# src/utils.py
import pickle
import os
import glob
import xarray as xr
import numpy as np
import yaml
from typing import Dict, List, Optional, Any, Tuple
from statsmodels.tsa.seasonal import STL

from scipy.signal import butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
from collections import defaultdict
from tqdm import tqdm
import matplotlib.cm as cm

# ============================================================
# 1. 掩膜分组 & 平铺工具
# ============================================================

# In src/utils.py

def standardize_mask(mask: np.ndarray) -> np.ndarray:
    """一个严格的掩码标准化函数，确保输出为布尔型。"""
    return np.ascontiguousarray(mask > 0, dtype=bool)

# --- 辅助函数：计算全球加权平均 (保持不变) ---
def _calculate_global_mean(data_array: xr.DataArray):
    """计算全球面积加权平均值。"""
    lat_coord_name = None
    if 'lat' in data_array.coords:
        lat_coord_name = 'lat'
    elif 'latitude' in data_array.coords:
        lat_coord_name = 'latitude'
    else:
        raise ValueError("Latitude coordinate ('lat' or 'latitude') not found.")

    weights = np.cos(np.deg2rad(data_array[lat_coord_name]))
    weights.name = "weights"

    spatial_dims = [d for d in data_array.dims if d not in ['time', 'bnds', 'time_bnds']]
    global_mean = data_array.weighted(weights).mean(spatial_dims)
    return global_mean


def calculate_vertical_mean(da_profile: xr.DataArray, depth_range: list) -> xr.DataArray:
    """
    对3D数据进行垂直加权平均 (weighted integration)。
    da_profile: (time, lev) or (time, lev, lat, lon)
    """
    # 1. 自动识别深度维度
    depth_dim = next((d for d in ['lev', 'olevel', 'depth'] if d in da_profile.dims), None)
    if not depth_dim:
        # 如果没有深度维度，假设已经是处理好的数据，直接返回
        return da_profile
    # 2. 截取深度范围
    # 假设深度单位是米，且坐标是数值型
    try:
        da_subset = da_profile.sel({depth_dim: slice(depth_range[0], depth_range[1])})
    except:
        # 有些模型深度坐标可能是反向的 (从大到小)，尝试反向切片
        da_subset = da_profile.sel({depth_dim: slice(depth_range[1], depth_range[0])})
    if da_subset.sizes[depth_dim] == 0:
        raise ValueError(f"No data found in depth range {depth_range}")
    # 3. 计算层厚权重
    depths = da_subset[depth_dim].values
    if len(depths) < 2:
        return da_subset.mean(dim=depth_dim)  # 只有一层
    # 简单估算层厚: dz[i] = (depth[i+1] - depth[i-1]) / 2
    # 边界处理: 第一层和最后一层
    bounds = np.concatenate([[depths[0]], (depths[:-1] + depths[1:]) / 2, [depths[-1]]])
    dz = np.diff(bounds)
    # 确保 dz 为正 (防止深度坐标反向)
    dz = np.abs(dz)
    weights = xr.DataArray(dz, coords={depth_dim: depths}, dims=depth_dim)
    # 4. 加权平均
    return da_subset.weighted(weights).mean(dim=depth_dim)


def find_corresponding_y_file(x_filepath: str, y_source_dir: str) -> str:
    """
    根据X的文件名，在Y的源目录中找到对应的文件。
    兼容带有 _3M, _12M 等后缀的 X 文件名。
    """
    filename = os.path.basename(x_filepath)

    # 【核心修正】: 去除 Rolling 后缀 (_3M.nc -> .nc)
    # 这样我们才能匹配到原始的 Y 文件名
    import re
    # 假设后缀是 _\d+M.nc
    original_filename = re.sub(r'_\d+M\.nc$', '.nc', filename)

    # 提取模型名 (用于构建子目录路径)
    # 对于 CMIP6: "MPI-ESM1-2-LR_r1i1p1f1.nc"
    if "_r" in original_filename:
        model_name = original_filename.split('_r')[0]
    else:
        # Fallback
        model_name = original_filename.split('_')[0]

    # 构造可能的 Y 文件名
    # 假设 Y 文件名加上了 _fldmean 后缀
    # 例如: CanESM5_r10i1p1f1.nc -> CanESM5_r10i1p1f1_fldmean.nc
    y_filename = original_filename.replace(".nc", "_fldmean.nc")

    # 构造完整路径
    # 假设目录结构是: y_source_dir / model_name / y_filename
    y_full_path = os.path.join(y_source_dir, model_name, y_filename)

    # 如果找不到，尝试不带子目录直接找
    if not os.path.exists(y_full_path):
        y_full_path_flat = os.path.join(y_source_dir, y_filename)
        if os.path.exists(y_full_path_flat):
            return y_full_path_flat

    # 如果还找不到，尝试递归搜索 (最稳健)
    if not os.path.exists(y_full_path):
        matches = glob.glob(os.path.join(y_source_dir, "**", y_filename), recursive=True)
        if matches:
            return matches[0]

    return y_full_path

def evaluate_reconstruction(y_true: np.ndarray, y_pred: np.ndarray, series_name: str):
    """
    计算并打印重建结果的评估指标 (RMSE 和 Correlation)。

    Args:
        y_true (np.ndarray): 真值时间序列。
        y_pred (np.ndarray): 重建/预测的时间序列。
        series_name (str): 用于在日志中标识当前评估的是哪个序列。
    """
    print(f"\n--- Evaluating: {series_name} ---")

    # 找到 y_true 和 y_pred 都没有 NaN 的位置
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if np.sum(valid_indices) < 2:
        print("  -> Not enough valid data points to calculate metrics.")
        return None

    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    # 1. 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))

    # 2. 计算 Pearson Correlation
    correlation = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]

    # 3. 计算 R-squared (决定系数)
    r_squared = correlation ** 2

    print(f"  -> Root Mean Square Error (RMSE): {rmse:.6f}")
    print(f"  -> Pearson Correlation (r):      {correlation:.4f}")
    print(f"  -> R-squared (r²):               {r_squared:.4f}")

    return {
        "rmse": rmse,
        "correlation": correlation,
        "r_squared": r_squared
    }


def reconstruct_timeseries(
        model_library: dict,
        hash_to_mask_map: dict,
        source_values_anom: np.ndarray,
        mask_values: np.ndarray,  # 需要掩码来查找
        source_name: str = "Unknown Source"
) -> np.ndarray:
    n_time = source_values_anom.shape[0]
    recon_anom = np.full(n_time, np.nan)

    print(f"\n--- Starting reconstruction for '{source_name}' (V2 Logic) ---")

    # 预先为每个时间步计算哈希
    time_hashes = [hash(standardize_mask(mask_values[t]).tobytes()) for t in range(n_time)]

    for mask_hash, pipeline in tqdm(model_library.items(), desc=f"Processing masks for {source_name}"):
        # 找到所有使用当前这个 mask_hash 的时间点
        time_indices = [i for i, h in enumerate(time_hashes) if h == mask_hash]

        if not time_indices:
            continue

        # 批量提取所有这些时间点的数据
        # (n_samples, lat, lon)
        batch_fields = source_values_anom[time_indices]

        # 批量提取特征向量
        mask_2d = hash_to_mask_map[mask_hash]
        # (n_samples, n_features)
        X_batch = np.array([np.nan_to_num(field[mask_2d]) for field in batch_fields])

        if X_batch.shape[0] > 0:
            # 批量预测
            predictions = pipeline.predict(X_batch)
            # 将预测结果放回正确的位置
            recon_anom[time_indices] = predictions

    return recon_anom


def calculate_uncertainty_weights(mask_da_values: np.ndarray, baseline_coverage: float = 0.10,
                                  smoothing_window: int = 60) -> np.ndarray:  # 新增 smoothing_window 参数
    """
    根据每个时间步的观测覆盖度，计算不确定性放大权重。
    【新增】加入了时间平滑，避免误差范围剧烈跳动。

    Args:
        smoothing_window: 平滑窗口大小（时间步数）。120表示10年（如果是月度数据）。
    """
    # 1. 计算每个时间步的有效格点数
    n_points_raw = np.sum(mask_da_values > 0, axis=(1, 2))

    # --- 对观测数量进行平滑 ---

    n_points_smooth = pd.Series(n_points_raw).rolling(
        window=smoothing_window, center=True, min_periods=1
    ).mean().values
    # 2. 计算总格点数
    total_ocean_points = np.sum(np.any(mask_da_values > 0, axis=0))
    if total_ocean_points == 0: total_possible_points = 1

    # 3. 计算覆盖度 (使用平滑后的点数)
    coverage = n_points_smooth / total_ocean_points

    # 4. 计算权重
    epsilon = 1e-6
    power = 1.5
    #这里定义时代因子：
    start_factor = 1.0
    end_factor = 0.1

    weights = np.sqrt(baseline_coverage / (coverage + epsilon))
    # weights = (baseline_coverage / (coverage + epsilon)) ** power
    era_factor = np.linspace(start_factor, end_factor, len(weights))
    weights = weights * era_factor
    # 5. 限制权重范围
    weights = np.clip(weights, a_min=0.1, a_max=8.0)

    return weights

def build_training_matrices(data: dict, min_features: int = 5) -> tuple:
    """
    从dataset.py返回的Numpy数组中，构建按mask_hash分组的训练矩阵。
    """
    print("\n--- [Step 8] Building training matrices from Numpy arrays ---")

    # 直接获取Numpy数组
    X_stacked_values = data["training_da_values"]  # (n_samples, lat, lon)
    Y_stacked_values = data["Y_train_truth_values"]  # (n_samples,)
    groups_stacked = data["training_groups"]  # (n_samples,)
    mask_stacked_values = data["training_mask_da_values"]  # (n_samples, lat, lon)
    #获取偏差集合
    inject_uncertainty = data.get("inject_uncertainty", False)
    bias_ensemble = data.get("bias_ensemble_values", None)
    print(f"   [DEBUG] inject_uncertainty set to: {data.get('inject_uncertainty')}") # <<<< 打印

    if inject_uncertainty and bias_ensemble is None:
        print(
            "[WARN] Uncertainty injection was requested but no bias ensemble was provided. Proceeding with clean training.")
        inject_uncertainty = False
    num_bias_members = bias_ensemble.shape[0] if inject_uncertainty else 0
    time_len = bias_ensemble.shape[1] if inject_uncertainty else 0
    print("time_len: ",time_len)
    print("num_bias_members: ",num_bias_members)
    n_samples = X_stacked_values.shape[0]
    print("n_samples: ",n_samples)

    # 2. 按掩码分组，并在过程中注入不确定性
    X_grouped, Y_grouped, groups_grouped = defaultdict(list), defaultdict(list), defaultdict(list)
    hash_to_mask_map = {}

    print("   -> Grouping samples by unique masks...")

    for i in tqdm(range(n_samples), desc="Grouping Samples"):
        mask_2d = standardize_mask(mask_stacked_values[i])

        # 第一次过滤：掩码本身必须有足够的点
        if np.sum(mask_2d) >= min_features:
            mask_hash = hash(mask_2d.tobytes())

            if mask_hash not in hash_to_mask_map:
                hash_to_mask_map[mask_hash] = mask_2d

            # (A) 提取干净的特征向量
            field_2d = X_stacked_values[i] #这里现在应该已经是稀疏伪观测场
            # x_vec = np.nan_to_num(field_2d[mask_2d])  # x_vec的长度现在 >= min_features
            x_vec = np.nan_to_num(field_2d[mask_2d])

            # (B) 核心注入逻辑
            if inject_uncertainty:
                bias_member_idx = np.random.randint(0, num_bias_members)
                time_idx = i % time_len
                bias_field_2d = bias_ensemble[bias_member_idx, time_idx, :, :]
                bias_vec = bias_field_2d[mask_2d]
                x_vec = x_vec + np.nan_to_num(bias_vec)
            # (C) 将最终的特征向量添加到组中
            X_grouped[mask_hash].append(x_vec)
            Y_grouped[mask_hash].append(Y_stacked_values[i])
            groups_grouped[mask_hash].append(groups_stacked[i])

    # 第二次过滤，在转换为Numpy数组后
    print("\n   -> Converting and filtering grouped lists...")

    final_X_grouped, final_Y_grouped, final_groups_grouped, final_hash_map  = {}, {}, {}, {}

    for h, x_list in tqdm(X_grouped.items(), desc="Filtering Groups"):
        X_array = np.array(x_list)

        # 检查特征数量是否一致 (防止极少数的错误)
        if X_array.ndim != 2 or X_array.shape[1] == 0:
            print(f"[WARN] Skipping mask {h} due to inconsistent feature dimensions.")
            continue

        # 检查训练样本数量是否足够进行交叉验证
        unique_groups_in_batch = np.unique(groups_grouped[h])
        if len(unique_groups_in_batch) < 2:
            print(
                f"[WARN] Skipping mask {h}: only {len(unique_groups_in_batch)} model(s) available, need at least 2 for CV.")
            continue

        # 如果所有检查都通过，才加入最终的训练列表
        final_X_grouped[h] = X_array
        final_Y_grouped[h] = np.array(Y_grouped[h])
        final_groups_grouped[h] = np.array(groups_grouped[h])
        final_hash_map[h] = hash_to_mask_map[h]


    print(f"   -> Found {len(final_X_grouped)} unique, valid masks for training.")

    return final_X_grouped, final_Y_grouped, final_groups_grouped, final_hash_map


def visualize_reconstruction_timeseries(
        time_coord: np.ndarray,
        save_path: str,
        title: str,
        ylabel: str = "Mean DO Anomaly",
        timeseries_dict: Optional[Dict[str, np.ndarray]] = None,
        colors_dict: Optional[Dict[str, str]] = None,
        is_annual_data: bool = False
):
    """
    一个通用的、灵活的时间序列可视化函数。
    - 能同时绘制月度和年度对比图。
    - 或者只绘制年度对比图（如果输入是年度数据）。
    - 自动处理NaN值，避免不正确的连线。
    - 支持自定义颜色。

    Args:
        time_coord (np.ndarray): 时间坐标数组 (datetime64 for monthly, years as int/float for annual)。
        save_path (str): 图像保存路径。
        title (str): 图像主标题。
        ylabel (str): Y轴标签。
        timeseries_dict (Optional[Dict]): 包含所有要绘制曲线的字典。
        colors_dict (Optional[Dict]): 为特定名称的曲线指定颜色的字典。
        is_annual_data (bool): 标记输入数据是否已经是年平均数据。
    """
    if timeseries_dict is None:
        timeseries_dict = {}

    # 过滤掉值全为nan的序列
    valid_timeseries = {k: v for k, v in timeseries_dict.items() if v is not None and not np.all(np.isnan(v))}
    if not valid_timeseries:
        print(f"[WARN] No valid data to plot for '{os.path.basename(save_path)}'. Skipping plot.")
        return

    # --- 准备颜色 ---
    if colors_dict is None:
        colors_dict = {}

    # 为没有指定颜色的曲线自动分配颜色
    num_series = len(valid_timeseries)
    auto_colors = cm.viridis(np.linspace(0, 1, num_series))

    final_colors = {}
    color_idx = 0
    for name in valid_timeseries.keys():
        if name in colors_dict:
            final_colors[name] = colors_dict[name]
        else:
            # 查找是否有名为 "My Reconstruction" 或 "Recon" 的变体
            if 'reconstruction' in name.lower() or 'recon' in name.lower():
                final_colors[name] = colors_dict.get("My Reconstruction", "red")
            else:
                final_colors[name] = auto_colors[color_idx]
                color_idx += 1

    # --- 绘图 ---
    plt.style.use('seaborn-v0_8-whitegrid')

    if is_annual_data:
        # 只绘制一张年度图
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.suptitle(title, fontsize=22, y=0.95)

        for name, series in valid_timeseries.items():
            series_pd = pd.Series(series, index=time_coord)
            series_clean = series_pd.dropna()
            ax.plot(series_clean.index, series_clean.values, label=name, color=final_colors.get(name), linewidth=2.0,
                    marker='o', markersize=4, alpha=0.9)

        ax.set_title('Annual Mean Comparison', fontsize=18)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle='--')
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)

    else:
        # 绘制月度和年度两张子图
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 15), sharex=False)
        fig.suptitle(title, fontsize=22, y=0.95)

        # --- 图一：月平均 ---
        ax1 = axes[0]
        df = pd.DataFrame(valid_timeseries, index=pd.to_datetime(time_coord))

        for name in df.columns:
            series_clean = df[name].dropna()
            ax1.plot(series_clean.index, series_clean.values, label=name, color=final_colors.get(name), linewidth=1.2,
                     alpha=0.9)

        ax1.set_title('Monthly Mean Comparison', fontsize=18)
        ax1.set_ylabel(ylabel, fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, which='both', linestyle='--')
        ax1.ticklabel_format(style='plain', axis='y', useOffset=False)

        # --- 图二：年平均 ---
        ax2 = axes[1]
        annual_df = df.resample('YE').mean()

        for name in annual_df.columns:
            series_annual_clean = annual_df[name].dropna()
            ax2.plot(series_annual_clean.index.year, series_annual_clean.values, label=name,
                     color=final_colors.get(name), linewidth=2.0, marker='o', markersize=4)

        ax2.set_title('Annual Mean Comparison', fontsize=18)
        ax2.set_xlabel('Year', fontsize=14)
        ax2.set_ylabel(ylabel, fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, which='both', linestyle='--')
        ax2.ticklabel_format(style='plain', axis='y', useOffset=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
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
        colors_dict: Optional[Dict[str, str]] = None
):
    """
    可视化带有不确定性范围（阴影）的重建结果，并可选择性地添加其他参考曲线。

    Args:
        time_coord (np.ndarray): 时间坐标数组。
        save_path (str): 图像保存路径。
        title (str): 图像主标题。
        ylabel (str): Y轴标签。
        median_line (np.ndarray): 中位线时间序列 (最佳估计)。
        lower_bound (np.ndarray): 不确定性范围的下界 (例如 2.5% 分位数)。
        upper_bound (np.ndarray): 不确定性范围的上界 (例如 97.5% 分位数)。
        other_lines_dict (Optional[Dict]): 一个字典，包含其他要绘制的参考曲线。
                                            键为曲线名称，值为时间序列数组。
        colors_dict (Optional[Dict]): 一个字典，为特定名称的曲线指定颜色。
    """
    print(f"\n--- Generating visualization with uncertainty: {os.path.basename(save_path)} ---")

    # --- 1. 准备数据 ---
    # 将所有数据放入一个 DataFrame 以便处理
    plot_data = {
        'Median': median_line,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
    }
    if other_lines_dict:
        plot_data.update(other_lines_dict)

    df = pd.DataFrame(plot_data, index=pd.to_datetime(time_coord))

    # 计算年平均数据
    annual_df = df.resample('YE').mean()

    # --- 2. 准备颜色 ---
    if colors_dict is None:
        colors_dict = {}

    # 为没有指定颜色的曲线自动分配颜色
    num_other_lines = len(other_lines_dict) if other_lines_dict else 0
    auto_colors = cm.viridis(np.linspace(0, 1, num_other_lines))
    color_idx = 0

    final_colors = {}
    if other_lines_dict:
        for name in other_lines_dict.keys():
            if name in colors_dict:
                final_colors[name] = colors_dict[name]
            else:
                final_colors[name] = auto_colors[color_idx]
                color_idx += 1

    # 为核心重建结果指定颜色
    final_colors['Median'] = colors_dict.get("My Reconstruction", "red")

    # --- 3. 绘图 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 15), sharex=False)
    fig.suptitle(title, fontsize=22, y=0.95)

    # --- 图一：月平均 ---
    ax1 = axes[0]

    # 绘制阴影区域
    ax1.fill_between(
        df.index, df['Lower Bound'], df['Upper Bound'],
        color=final_colors['Median'],
        alpha=0.2,
        label='95% Uncertainty Range'
    )
    # 绘制中位线
    df_median_clean = df['Median'].dropna()
    ax1.plot(df_median_clean.index, df_median_clean.values, color=final_colors['Median'],
             label='My Reconstruction (Median)', linewidth=1.5)

    # 绘制其他对比曲线
    if other_lines_dict:
        for name in other_lines_dict.keys():
            df_other_clean = df[name].dropna()
            ax1.plot(df_other_clean.index, df_other_clean.values, label=name, color=final_colors[name], linewidth=1.2,
                     alpha=0.9)

    ax1.set_title('Monthly Mean Comparison', fontsize=18)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--')
    ax1.ticklabel_format(style='plain', axis='y', useOffset=False)

    # --- 图二：年平均 ---
    ax2 = axes[1]

    # 绘制阴影区域
    ax2.fill_between(
        annual_df.index.year, annual_df['Lower Bound'], annual_df['Upper Bound'],
        color=final_colors['Median'],
        alpha=0.2,
        label='95% Uncertainty Range'
    )
    # 绘制中位线
    annual_df_median_clean = annual_df['Median'].dropna()
    ax2.plot(annual_df_median_clean.index.year, annual_df_median_clean.values, color=final_colors['Median'],
             label='My Reconstruction (Median)', linewidth=2.0, marker='o', markersize=4)

    # 绘制其他对比曲线
    if other_lines_dict:
        for name in other_lines_dict.keys():
            annual_df_other_clean = annual_df[name].dropna()
            ax2.plot(annual_df_other_clean.index.year, annual_df_other_clean.values, label=name,
                     color=final_colors[name], linewidth=1.5, marker='.', markersize=4, alpha=0.9)

    ax2.set_title('Annual Mean Comparison', fontsize=18)
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, which='both', linestyle='--')
    ax2.ticklabel_format(style='plain', axis='y', useOffset=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Visualization with uncertainty saved to: {save_path}")
    plt.close(fig)


def apply_filter(
        series: np.ndarray,
        cutoff_period: float,
        filter_type: str,
        sample_rate: float = 1.0  # 假设是年度数据，每年采样1次
) -> np.ndarray:
    """
    对时间序列应用一个巴特沃斯滤波器 (低通或高通)。

    Args:
        series (np.ndarray): 输入的一维时间序列。
        cutoff_period (float): 截止周期 (单位与采样率一致，例如 "年")。
        filter_type (str): 'low' (低通) 或 'high' (高通)。
        sample_rate (float): 数据的采样频率。对于年度数据是1.0，月度数据是12.0。

    Returns:
        np.ndarray: 滤波后的时间序列。
    """
    # 检查输入
    if filter_type not in ['low', 'high']:
        raise ValueError("filter_type must be 'low' or 'high'")

    # 处理NaN值：线性插值填充内部的NaN，但保留开头和结尾的NaN
    series_pd = pd.Series(series)
    series_interpolated = series_pd.interpolate(method='linear', limit_direction='both').values
    nan_mask = np.isnan(series)  # 记住原始NaN的位置

    # 计算奈奎斯特频率和截止频率
    nyquist_freq = 0.5 * sample_rate
    cutoff_freq = 1.0 / cutoff_period

    # 归一化截止频率
    normalized_cutoff = cutoff_freq / nyquist_freq

    # 设计滤波器 (4阶巴特沃斯)
    b, a = butter(N=4, Wn=normalized_cutoff, btype=filter_type)

    # 应用滤波器 (filtfilt 进行零相位滤波，避免时间延迟)
    filtered_series = filtfilt(b, a, series_interpolated)

    # 将原始的NaN位置恢复回去
    filtered_series[nan_mask] = np.nan

    return filtered_series


def save_results_to_parquet(
        filepath: str,
        time_coord: np.ndarray,
        results_dict: dict,
        config: dict
):
    """
    将所有实验结果和元数据保存到一个Parquet文件中。

    Args:
        filepath (str): 输出的Parquet文件路径。
        time_coord (np.ndarray): 实验使用的时间坐标。
        results_dict (dict): 包含所有时间序列的字典。
        config (dict): 实验的配置字典，用于记录元数据。
    """
    print(f"\n--- Saving experiment results to: {os.path.basename(filepath)} ---")

    # --- 1. 创建 DataFrame ---
    # 使用时间坐标作为索引
    df = pd.DataFrame(results_dict, index=pd.to_datetime(time_coord))

    # --- 2. 添加元数据 ---
    # 将配置信息平铺，以便存储
    # 我们只保存 'data' 和 'experiment' 部分
    metadata = {
        'data_config': str(config.get('data', {})),
        'experiment_config': str(config.get('experiment', {}))
    }

    # --- 3. 保存 ---
    # pandas 的 to_parquet 允许我们嵌入元数据
    # 但为了更通用，我们也可以将元数据作为额外的列或单独的文件
    # 一个简单的方法是创建一个 MultiIndex 列

    # 我们将元数据直接保存到 Parquet 文件的 metadata 字段中
    # 这需要 pyarrow 引擎
    try:
        df.to_parquet(filepath, engine='pyarrow', compression='gzip')

        # 将元数据保存为一个关联的 .yaml 文件，更易读
        meta_filepath = filepath.replace('.parquet', '_meta.yaml')
        with open(meta_filepath, 'w') as f:
            yaml.dump(config, f)

        print(f"   -> Data saved to {filepath}")
        print(f"   -> Metadata saved to {meta_filepath}")

    except Exception as e:
        print(f"[ERROR] Failed to save results to Parquet. Error: {e}")
        print("        Please ensure 'pyarrow' is installed (`pip install pyarrow`).")


def load_results_from_parquet(filepath: str) -> tuple:
    """
    从Parquet文件和关联的元数据文件中加载实验结果。

    Returns:
        tuple: (pd.DataFrame, dict) 包含结果数据和配置元数据。
    """
    print(f"\n--- Loading experiment results from: {os.path.basename(filepath)} ---")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")

    # 读取 Parquet 文件
    df = pd.read_parquet(filepath)

    # 读取关联的元数据 YAML 文件
    meta_filepath = filepath.replace('.parquet', '_meta.yaml')
    config = {}
    if os.path.exists(meta_filepath):
        with open(meta_filepath, 'r') as f:
            config = yaml.safe_load(f)

    print("   -> Results loaded successfully.")
    return df, config



def build_mask_groups(mask_da: xr.DataArray):
    """
    根据 obs_mask_da 在 time 维度上的观测模式，将时间步分组。

    返回:
        hash_to_mask_map: dict[hash] -> {
            "mask_2d": (lat, lon) 的 bool 掩膜，代表这一组观测网络的 pattern,
            "time_indices":  属于该组的所有时间索引 list[int]
        }
        groups_grouped: dict[hash] -> np.ndarray  (这里暂时不用也可以)
    """
    if "time" not in mask_da.dims:
        # 没有时间维，就当只有一个 group
        mask_2d = (mask_da > 0)
        hash_to_mask_map = {
            "single": {
                "mask_2d": mask_2d,
                "time_indices": None,
            }
        }
        groups_grouped = {"single": np.array([0])}
        return hash_to_mask_map, groups_grouped

    # 确保 lat/lon 名称统一
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
        m_t = (da.isel(time=t) > 0)  # (lat, lon)
        # 转成 1D bool，并 hash
        pattern = m_t.values.astype(np.uint8).flatten()
        h = hash(pattern.tobytes())

        if h not in hash_to_mask_map:
            # 第一次遇到这个观测网络模式
            hash_to_mask_map[h] = {
                "mask_2d": m_t,        # 直接存这个 2D 掩膜 (lat, lon)
                "time_indices": [t],   # 初始化时间索引
            }
            groups_grouped[h] = np.array([t])
        else:
            hash_to_mask_map[h]["time_indices"].append(t)
            groups_grouped[h] = np.append(groups_grouped[h], t)

    return hash_to_mask_map, groups_grouped


def flatten_field_with_mask(field_2d: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    """
    将 (lat, lon) 场和 (lat, lon) 掩膜展平成一维向量，只保留 mask>0 的点。
    容错：
      - 如果收到 (1, H, W) 之类的 3D，就自动 squeeze 掉长度为 1 的轴。
      - 如果 mask 是 3D，但其中某个轴长度为 1，同样 squeeze。
    """
    field = np.asarray(field_2d)
    mask = np.asarray(mask_2d)

    # 自动 squeeze 掉长度为 1 的轴
    if field.ndim == 3 and 1 in field.shape:
        field = np.squeeze(field)
    if mask.ndim == 3 and 1 in mask.shape:
        mask = np.squeeze(mask)

    if field.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"flatten_field_with_mask expects 2D arrays after squeeze, "
            f"got field shape {field.shape}, mask shape {mask.shape}"
        )

    if field.shape != mask.shape:
        raise ValueError(f"Shape mismatch: field {field.shape}, mask {mask.shape}")

    valid = mask > 0
    return field[valid]

def ensure_training_anomaly(data: dict):
    """
    确保返回一个 anomaly 形式的 training_da_anom: (model, time, lat, lon)
    优先：
      - 如果 data 中已经有 'training_da_anom'，直接用。
    否则：
      - 如果有 'training_da'，则按 time 维做 anomaly：减去 time 平均。
      - 否则抛出错误，请在 dataset.py 中添加相应字段。
    """
    if "training_da_anom" in data:
        training_da_anom = data["training_da_anom"]
        print("[INFO] Using 'training_da_anom' from dataset.load_and_prepare_data().")
        return training_da_anom

    if "training_da" in data:
        training_da = data["training_da"]
        print("[INFO] 'training_da_anom' not found, building anomaly from 'training_da' (subtract time mean).")
        # training_da: (model, time, lat, lon)
        clim = training_da.mean(dim="time")
        training_da_anom = training_da - clim
        return training_da_anom

    raise KeyError(
        "Neither 'training_da_anom' nor 'training_da' found in data dict. "
        "Please check dataset.load_and_prepare_data() to ensure it returns "
        "either 'training_da_anom' or (at least) 'training_da'."
    )

# ============================================================
# 2. 区域平均（支持 anomaly & 绝对值）
# ============================================================

def compute_regional_mean_timeseries(
    da: xr.DataArray,
    mask_da: xr.DataArray = None,
) -> np.ndarray:
    """
    计算区域平均时间序列（或单一标量）：
        - da: 可以是
            (time, lat, lon)  -> 返回 shape (time,)
            (model, time, lat, lon) -> 返回 shape (model, time)
            (lat, lon) -> 返回标量
        - mask_da:
            (time, lat, lon) 或 (lat, lon)
            >0 为有效点；若为 None，则对所有点平均。
    区域平均默认按纬向 cos(lat) 加权（非常接近面积权重）。
    """

    # 统一 lat / lon 名称
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
            # 掩膜有 time，但场没有 -> 使用时间平均的掩膜
            m = m.mean("time") > 0
        if "time" in da.dims and "time" not in m.dims:
            # 场有 time，但掩膜没有 -> broadcast
            m = m.broadcast_like(da.isel(time=0))
        da = da.where(m > 0)

    # 纬向权重
    if "lat" in da.dims:
        lat_rad = np.deg2rad(da["lat"])
        weights = np.cos(lat_rad)
        weights = weights / weights.mean()  # 归一，避免过大过小
    else:
        weights = None

    if "lat" in da.dims and "lon" in da.dims:
        # 对 lat/lon 做面积加权平均
        if "model" in da.dims and "time" in da.dims:
            # (model, time, lat, lon)
            # 先对 lon 平均
            da_lon = da.weighted(None if weights is None else weights).mean(dim=("lat", "lon"))
            y = da_lon.values  # (model, time)
        elif "time" in da.dims:
            da_reg = da.weighted(None if weights is None else weights).mean(dim=("lat", "lon"))
            y = da_reg.values  # (time,)
        else:
            # (lat, lon) -> 单一标量
            da_reg = da.weighted(None if weights is None else weights).mean(dim=("lat", "lon"))
            y = float(da_reg.values)
    else:
        # 没有经纬度，就直接平均所有维
        da_reg = da.mean()
        y = da_reg.values

    return y


# ============================================================
# 3. 评估函数（RMSE / corr）
# ============================================================

def evaluate_reconstruction(y_true: np.ndarray, y_pred: np.ndarray, series_name: str):
    """
    计算并打印重建结果的评估指标 (RMSE 和 Correlation)。

    Args:
        y_true (np.ndarray): 真值时间序列。
        y_pred (np.ndarray): 重建/预测的时间序列。
        series_name (str): 用于在日志中标识当前评估的是哪个序列。
    """
    print(f"\n--- Evaluating: {series_name} ---")

    # 找到 y_true 和 y_pred 都没有 NaN 的位置
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if np.sum(valid_indices) < 2:
        print("  -> Not enough valid data points to calculate metrics.")
        return {"rmse": np.nan, "correlation": np.nan, "r_squared": np.nan}

    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    # 1. 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))

    # 2. 计算 Pearson Correlation
    correlation = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]

    # 3. 计算 R-squared (决定系数)
    r2 = r2_score(y_true_valid, y_pred_valid)

    print(f"  -> Root Mean Square Error (RMSE): {rmse:.6f}")
    print(f"  -> Pearson Correlation (r):      {correlation:.4f}")
    print(f"  -> R-squared (r²):               {r2:.4f}")

    return {
        "rmse": rmse,
        "correlation": correlation,
        "r_squared": r2
    }


# ============================================================
# 4. 可视化：月度 + 年度平均曲线
# ============================================================

def _monthly_to_annual(time: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    把按月的时间序列聚合为按年的平均。
        time:   numpy.datetime64 数组，长度 T
        values: 同长度的一维数组
    返回：
        years:  年份数组
        annual: 每年平均
    """
    # 提取年份
    years = np.array([t.astype("datetime64[Y]").astype(int) + 1970 for t in time])
    unique_years = np.unique(years)

    annual_vals = []
    for y in unique_years:
        mask = years == y
        if mask.sum() == 0:
            annual_vals.append(np.nan)
        else:
            annual_vals.append(np.nanmean(values[mask]))
    return unique_years, np.array(annual_vals)

# ============================================================
# 5. 可视化 betas（简单版本）
# ============================================================

def visualize_betas(
    model_library: Dict[int, Any],
    hash_to_mask_map: Dict[int, Dict[str, Any]],
    data: Dict[str, Any],
    output_dir: str,
    max_plots: int = 6,
):
    """
    简单地把若干个掩膜组的 beta 映射回 2D 场，画在地图上。
    约定：
        - model_library[mask_hash] 是一个 dict，里头有 'coef' (np.ndarray: shape (n_features,)).
        - hash_to_mask_map[mask_hash]['mask_2d'] 是 (lat, lon) 的布尔掩膜。
        - data 中包含一个参考纬度 / 经度的 DataArray，用来确定 lat/lon 网格。
          这里默认用 data["observation_source_model_da"]。
    """
    if len(model_library) == 0:
        print("No models in model_library, skip visualize_betas.")
        return

    ref_da = data.get("observation_source_model_da", None)
    if ref_da is None:
        print("No reference field in data for visualize_betas, skip.")
        return

    # 统一 lat/lon 名称
    rename = {}
    if "longitude" in ref_da.dims:
        rename["longitude"] = "lon"
    if "latitude" in ref_da.dims:
        rename["latitude"] = "lat"
    if rename:
        ref_da = ref_da.rename(rename)

    lat = ref_da["lat"].values
    lon = ref_da["lon"].values
    ny = len(lat)
    nx = len(lon)

    betas_dir = os.path.join(output_dir, "betas_maps")
    os.makedirs(betas_dir, exist_ok=True)

    # 只画前 max_plots 个掩膜组，防止太多
    for i, (mask_hash, model_info) in enumerate(model_library.items()):
        if i >= max_plots:
            break

        coef = model_info.get("coef", None)
        if coef is None:
            continue

        mask_info = hash_to_mask_map.get(mask_hash, None)
        if mask_info is None:
            continue
        mask_2d = mask_info["mask_2d"]
        if mask_2d.shape != (ny, nx):
            # 尝试对齐一下
            try:
                mask_2d = mask_2d.reindex_like(ref_da.isel(time=0), method="nearest")
                mask_2d = mask_2d.values
            except Exception:
                print(f"Mask shape mismatch for hash {mask_hash}, skip.")
                continue
        else:
            mask_2d = mask_2d.values

        # 把 coef 填回 (lat, lon) 网格
        beta_map = np.full((ny, nx), np.nan, dtype=float)
        beta_map[mask_2d] = coef

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.pcolormesh(lon, lat, beta_map, shading="auto")
        ax.set_title(f"Regression Coefficients (mask_hash={mask_hash})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Beta")
        fig.tight_layout()

        fig_path = os.path.join(betas_dir, f"betas_mask_{i}_hash_{mask_hash}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Beta maps saved to: {betas_dir}")


def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算一组序列的详细统计指标。

    Args:
        y_true: 真值序列 (Obs/Model Truth)
        y_pred: 重建序列 (Reconstruction)

    Returns:
        dict: 包含 Bias, RMSE, Corr, Amp_Error, AAE 的字典
    """
    # 移除 NaN 以进行统计计算
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    t = y_true[mask]
    p = y_pred[mask]

    if len(t) < 2:
        return {k: np.nan for k in ['Bias', 'RMSE', 'Corr', 'Amp_Error', 'AAE']}

    # 1. Bias (偏差): 正值代表重建偏高，负值代表偏低
    bias = np.mean(p - t)

    # 2. RMSE (均方根误差): 综合反映误差大小
    rmse = np.sqrt(mean_squared_error(t, p))

    # 3. Correlation (相关系数 r): 反映相位一致性
    corr = np.corrcoef(t, p)[0, 1]

    # 4. Amplitude Error (标准差百分比误差): 核心指标
    # 反映重建结果是否丢失了变率 (Dampening) 或者高估了变率
    std_true = np.std(t, ddof=1)
    std_pred = np.std(p, ddof=1)
    # 如果真值几乎是常数，防止除以零
    if std_true < 1e-9:
        amp_error = np.nan
    else:
        amp_error = ((std_pred - std_true) / std_true) * 100

    # 5. AAE (平均绝对误差): 对离群值不如 RMSE 敏感
    aae = np.mean(np.abs(p - t))

    return {
        'Bias': bias,
        'RMSE': rmse,
        'Corr': corr,
        'Amp_Error (%)': amp_error,
        'AAE': aae
    }


def comprehensive_evaluation(
        time_coord,
        y_true_raw,
        y_pred_raw,
        output_dir,
        var_name="DO",
        trend_window=121,  # 约10年 (奇数)
        seasonal_window=13
):
    """
    执行全套评估流程：STL分解 -> 分量指标计算 -> 绘图。

    Args:
        time_coord: 时间坐标数组 (datetime64)
        y_true_raw: 真值数组 (numpy array)
        y_pred_raw: 重建值数组 (numpy array)
        output_dir: 结果保存路径
        trend_window: STL分解中趋势项的窗口长度 (月数)
    """
    print(f"\n--- Starting Comprehensive Evaluation with STL Decomposition ---")

    # 1. 数据预处理：转换为 Pandas Series 并处理时间索引
    # STL 需要明确的频率 (Frequency)，这里假设是月度 'MS' (Month Start)
    ts_index = pd.to_datetime(time_coord)
    s_true = pd.Series(y_true_raw, index=ts_index)
    s_pred = pd.Series(y_pred_raw, index=ts_index)

    # 简单的线性插值处理 NaN，因为 STL 不支持 NaN 输入
    # 注意：只用于分解，计算指标时我们会匹配原始掩码
    s_true_filled = s_true.interpolate(method='linear', limit_direction='both')
    s_pred_filled = s_pred.interpolate(method='linear', limit_direction='both')

    # 2. STL 时间尺度分解
    # period=12 对应年周期
    print("   -> Performing STL Decomposition...")

    def decompose(series):
        res = STL(series, period=12, trend=trend_window, seasonal=seasonal_window).fit()
        return res.trend, res.seasonal, res.resid

    true_trend, true_season, true_resid = decompose(s_true_filled)
    pred_trend, pred_season, pred_resid = decompose(s_pred_filled)

    # 组装分量字典
    # Original: 原始数据
    # Decadal: 长期趋势 (Trend)
    # Seasonal: 季节循环
    # Sub-decadal: 残差项 (Residual = Original - Trend - Seasonal)，代表高频变率
    components = {
        'Original': (s_true, s_pred),  # 使用未插值的原始数据计算指标
        'Decadal (Trend)': (true_trend, pred_trend),
        'Seasonal': (true_season, pred_season),
        'Sub-decadal (Resid)': (true_resid, pred_resid)
    }

    # 3. 计算统计指标 (Metrics)
    results_list = []
    print("   -> Calculating Metrics for each component...")

    for comp_name, (t_series, p_series) in components.items():
        metrics = calculate_detailed_metrics(t_series.values, p_series.values)
        metrics['Component'] = comp_name
        results_list.append(metrics)

    df_metrics = pd.DataFrame(results_list).set_index('Component')

    # 保存指标表
    csv_path = f"{output_dir}/evaluation_metrics_decomposition.csv"
    df_metrics.to_csv(csv_path)
    print(f"   -> Metrics saved to {csv_path}")
    print(df_metrics)

    # 4. 绘图 1: 四分量对比图
    print("   -> Generating Component Comparison Plot...")
    fig1, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    axes = axes.flatten()

    colors = {'Truth': 'black', 'Recon': 'red'}

    for i, (comp_name, (t_series, p_series)) in enumerate(components.items()):
        ax = axes[i]
        ax.plot(t_series.index, t_series, color=colors['Truth'], label='Truth', linewidth=1.2, alpha=0.8)
        ax.plot(p_series.index, p_series, color=colors['Recon'], label='Reconstruction', linewidth=1.2, alpha=0.9,
                linestyle='--')

        # 在标题中展示关键指标
        metrics = df_metrics.loc[comp_name]
        title_str = (f"{comp_name}\n"
                     f"r = {metrics['Corr']:.3f} | "
                     f"Amp.Err = {metrics['Amp_Error (%)']:.1f}% | "
                     f"RMSE = {metrics['RMSE']:.4f}")
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0: ax.legend(loc='upper right')
        if i >= 2: ax.set_xlabel('Year')
        ax.set_ylabel(f'{var_name} (mol m-3)')

    plt.tight_layout()
    fig1.savefig(f"{output_dir}/plot_1_decomposition_comparison.png", dpi=300)
    plt.show()

    plt.close(fig1)

    # 5. 计算并绘图 2: 滑动 RMSE (Time-dependent Error)
    print("   -> Generating Rolling RMSE Plot...")

    # 窗口大小 k=12 (12个月)
    window_k = 12

    # 计算平方误差序列
    squared_error = (s_true - s_pred) ** 2

    # 滑动计算均值，然后开根号
    # center=True 保证误差对应到窗口中心年份
    rolling_rmse = np.sqrt(squared_error.rolling(window=window_k, center=True, min_periods=6).mean())

    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_rmse.index, rolling_rmse, color='darkorange', linewidth=2, label=f'{window_k}-Month Rolling RMSE')

    # 添加平均RMSE参考线
    avg_rmse = df_metrics.loc['Original', 'RMSE']
    ax.axhline(avg_rmse, color='gray', linestyle='--', label=f'Mean RMSE ({avg_rmse:.4f})')

    ax.set_title('Time-dependent Reconstruction Error (Rolling RMSE)', fontsize=14)
    ax.set_ylabel('RMSE (mol m-3)', fontsize=12)
    ax.set_xlabel('Year', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # 如果有明显的数据稀缺时段（如二战前），可以用背景色标出
    # ax.axvspan(pd.Timestamp('1900-01-01'), pd.Timestamp('1950-01-01'), color='gray', alpha=0.1, label='Data Sparse Era')

    plt.tight_layout()
    fig2.savefig(f"{output_dir}/plot_2_rolling_rmse.png", dpi=300)
    plt.show()
    plt.close(fig2)

    return df_metrics
