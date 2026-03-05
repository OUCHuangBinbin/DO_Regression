# src/dataset.py

import os
import glob
import xarray as xr
import numpy as np
from typing import Dict
time_slice = None
from Oxy_regression.src.utils import calculate_vertical_mean, find_corresponding_y_file

def calculate_weighted_mean(data_array: xr.DataArray) -> xr.DataArray:
    """
    计算给定DataArray的全球或区域面积加权平均值。
    这是计算地球物理场平均值的物理正确方法。
    """
    # 确保 data_array 是 xarray.DataArray
    if not hasattr(data_array, 'coords'):
        # 如果是numpy数组，无法加权，返回算术平均并警告
        print("[WARN] Input to calculate_weighted_mean is a Numpy array. Returning unweighted mean.")
        return np.nanmean(data_array, axis=tuple(range(1, data_array.ndim)))

    if 'lat' not in data_array.coords:
        raise ValueError("Latitude coordinate 'lat' not found for weighted mean calculation.")

    weights = np.cos(np.deg2rad(data_array.lat))
    weights.name = "weights"

    spatial_dims = [d for d in ['lon', 'lat'] if d in data_array.dims]
    if not spatial_dims:
        return data_array  # 已经是时间序列

    weighted_mean = data_array.weighted(weights).mean(spatial_dims)
    return weighted_mean

# --- 辅助函数：加载和预处理深度段均值文件 ---
def _load_and_process_target_y(
        y_path: str,
        var_name: str,
        time_slice: slice,
        depth_range: list
) -> xr.DataArray:
    """
    加载目标变量 Y (3D fldmean)，进行时间切片和垂直积分。
    """
    if not os.path.exists(y_path):
        return None

    try:
        with xr.open_dataset(y_path, decode_times=False) as ds:
            # 时间解码
            ds = xr.decode_cf(ds, use_cftime=True)
            if isinstance(ds.time.to_index(), xr.coding.cftimeindex.CFTimeIndex):
                ds['time'] = ds.indexes['time'].to_datetimeindex(unsafe=True)
            # 提取变量
            da = ds[var_name]
            # 时间切片
            da = da.sel(time=time_slice)
            if da.time.size == 0: return None
            # 垂直积分 ，这里计算平均值
            y_val = calculate_vertical_mean(da, depth_range)
            return y_val.squeeze(drop=True).load()

    except Exception as e:
        print(f"[WARN] Failed to load Y file {os.path.basename(y_path)}: {e}")
        return None


def find_corresponding_full_file(sparse_file_path, full_source_dir):
    """
    根据稀疏文件名 (如 Model_r1i1p1f1_3M.nc) 找到对应的完整文件名 (Model_r1i1p1f1.nc)。
    """
    filename = os.path.basename(sparse_file_path)

    # 1. 去除滚动窗口后缀 (_3M, _6M, _12M, _60M)
    # 假设后缀格式是 _XM.nc
    import re
    # 正则替换：将 _\d+M.nc 替换为 .nc
    full_filename = re.sub(r'_\d+M\.nc$', '.nc', filename)
    if "_r" in full_filename:
        model_name = full_filename.split("_r")[0]
        # 尝试构建完整路径
        potential_path = os.path.join(full_source_dir, model_name, full_filename)
        if os.path.exists(potential_path):
            return potential_path
    matches = glob.glob(os.path.join(full_source_dir, "**", full_filename), recursive=True)

    if not matches:
        # Debug info
        print(f"[DEBUG] Looking for: {full_filename}")
        print(f"[DEBUG] In directory: {full_source_dir}")
        return None

    return matches[0]

# --- 辅助函数：加载和预处理单个文件 ---
def _load_and_process_file(file_path: str, var_name: str,
                           depth_level: int, time_slice: slice,
                           is_obs_source: bool = False,
                           # target_resolution: Optional[int] = None
                           ) -> xr.DataArray:
    print(f"  -> Opening file: {os.path.basename(file_path)}")
    try:
        with xr.open_dataset(file_path, chunks=None, decode_times=False) as ds:
            ds = xr.decode_cf(ds, use_cftime=True)
            if isinstance(ds.time.to_index(), xr.coding.cftimeindex.CFTimeIndex):
                ds['time'] = ds.indexes['time'].to_datetimeindex(unsafe=True)

            ds_sliced = ds.sel(time=time_slice)
            if ds_sliced.time.size == 0:
                print(f"     ... WARNING: No data in specified time range for {os.path.basename(file_path)}. Skipping.")
                return None

            data_var = ds_sliced[var_name]
            data_var = data_var.where(data_var >= 0)
            # data_var = data_var.where(data_var > 1e-9)
            depth_dim_name = next((d for d in ['lev', 'olevel', 'depth'] if d in data_var.dims), None)

            if depth_dim_name and data_var.sizes[depth_dim_name] > 1:
                surface_data = data_var.isel({depth_dim_name: depth_level})
            else:
                surface_data = data_var

            dims_to_drop = [d for d in ['lev', 'olevel', 'depth'] if
                            d in surface_data.dims and surface_data.sizes[d] == 1]
            coords_to_drop = [d for d in ['lev', 'olevel', 'depth'] if d in surface_data.coords]

            if dims_to_drop:
                surface_data = surface_data.squeeze(dim=dims_to_drop, drop=True)
            if coords_to_drop:
                surface_data = surface_data.drop_vars(coords_to_drop, errors='ignore')

            #对实测数据进行一部分筛选，主要是检测离群值
            if is_obs_source:
                print("     ... Applying quality control for true observation source.")
                # 1. 过滤非物理值
                surface_data = surface_data.where(surface_data > 0)

                # --- 2. 过滤极端离群值 (Z-score method) ---
                # 计算每个格点在时间上的均值和标准差
                mean_over_time = surface_data.mean(dim='time')
                std_over_time = surface_data.std(dim='time')

                # 定义阈值，这里我假设的是超过4个标准差就被认为是离群值

                z_score_threshold = 4.0
                #之前做海表实验用的是4倍标准差的阈值，但是后面做深度段的溶解氧实验后数据的不确定性波动是很大的，现在收紧这个阈值。
                # z_score_threshold = 3.0
                # z_score_threshold = 2.0

                # 计算上限和下限
                upper_limit = mean_over_time + z_score_threshold * std_over_time
                lower_limit = mean_over_time - z_score_threshold * std_over_time
                # 只保留 0.1% 到 99.9% 之间的数据。
                lower_quantile = surface_data.quantile(0.001, skipna=True)
                upper_quantile = surface_data.quantile(0.999, skipna=True)
                print(
                    f"     ... Data range check: 0.1% quantile = {lower_quantile.item():.4f}, 99.9% quantile = {upper_quantile.item():.4f}")

                # 保留在 limit 之间的数据，否则设为 NaN
                original_count = surface_data.count().item()
                surface_data = surface_data.where(
                    (surface_data >= lower_limit) & (surface_data <= upper_limit)
                )
                filtered_count1 = surface_data.count().item()
                print(f" Firt QC[1]... Outlier filtering removed {original_count - filtered_count1} data points.")

                surface_data = surface_data.where(
                    (surface_data >= lower_quantile) & (surface_data <= upper_quantile)
                )
                filtered_count2 = surface_data.count().item()
                print(f" Second QC[2]... Outlier filtering removed {original_count - filtered_count2} data points.")

            return surface_data
    except Exception as e:
        print(f"     ... WARNING: Failed to process {os.path.basename(file_path)}. Error: {e}. Skipping.")
        return None

# --- 主函数：加载和准备所有数据 ---
def load_and_prepare_data(config: Dict) -> Dict:
    """
    Clean and physically consistent data loader.

    X : 2D (lat, lon) anomaly fields
    Y : depth-averaged + spatially averaged 1D anomaly time series
    """

    data_cfg = config["data"]

    # ------------------------------------------------------------------
    # Basic configuration
    # ------------------------------------------------------------------
    time_slice = slice(f"{data_cfg['time_range'][0]}-01-01",f"{data_cfg['time_range'][1]}-12-31")
    ref_slice = slice(f"{data_cfg['anomaly_reference_period'][0]}-01-01",f"{data_cfg['anomaly_reference_period'][1]}-12-31")
    varname = data_cfg.get("variable_name", "o2")
    input_depth = data_cfg.get('input_depth_level', 0)
    print(f"--- Enforcing Input Depth Level: {input_depth} (Surface) ---")
    # depth_level = data_cfg["depth_level"]
    y_cfg = data_cfg["target_y_config"]
    data_cfg = config['data']
    print(f"--- Time range enforced: {time_slice.start} → {time_slice.stop} ---")

    # ------------------------------------------------------------------
    # Step 1. Collect training model files
    # ------------------------------------------------------------------
    training_dirs = data_cfg.get("training_model_dirs", [])
    all_files = []
    for d in training_dirs:
        all_files.extend(glob.glob(os.path.join(d, "**", "*.nc"), recursive=True))

    all_files = sorted(set(all_files))

    exclude_paths = {
        # os.path.abspath(data_cfg["observation_source_model_path"]),
        os.path.abspath(data_cfg["observation_source_true_path"]),
    }

    training_model_files = [
        f for f in all_files if os.path.abspath(f) not in exclude_paths
    ]

    if not training_model_files:
        raise ValueError("No training model files found.")

    # ------------------------------------------------------------------
    # Step 2. Load paired (X, Y) training samples
    # ------------------------------------------------------------------

    true_obs_da = _load_and_process_file(
        data_cfg["observation_source_true_path"],
        data_cfg.get("true_obs_variable_name", varname),
        input_depth,
        time_slice,
        is_obs_source=True,
    )

    X_train_list = []
    Y_train_list = []
    full_model_dir = data_cfg.get("original_full_model_dir")
    print("\n--- Loading paired X–Y training samples ---")

    for f in training_model_files:
        # X: surface anomaly field
        x_da = _load_and_process_file(f,varname,input_depth,time_slice,is_obs_source=False,)
        if x_da is None: continue
        if full_model_dir:
            f_full = find_corresponding_full_file(f, full_model_dir)
            if not f_full:
                print(f"[WARN] Full file not found for {os.path.basename(f)}. Skipping.")
                continue
            x_da_full = _load_and_process_file(f_full, varname, input_depth, time_slice, is_obs_source=False)
        # 这里使用完整数据算气候态
            x_clim = x_da_full.sel(time=ref_slice).groupby("time.month").mean("time")
        else:
            # 如果没配置完整路径，回退到旧逻辑（在你的Rolling场景下会出错！）
            print("[WARN] No full model dir provided! Calculating climatology from sparse data (DANGEROUS!).")
            x_clim = x_da.sel(time=ref_slice).groupby("time.month").mean("time")
        # Y: depth-averaged truth
        y_path = find_corresponding_y_file(f, y_cfg["source_dir"])
        y_da = _load_and_process_target_y(y_path,varname,time_slice,y_cfg["depth_range"],)

        if y_da is None:continue

        if len(x_da.time) != len(y_da.time):
            print(f"[WARN] Time mismatch in {os.path.basename(f)}, skipped.")
            continue

        # --- anomalies ---
        # x_clim = x_da.sel(time=ref_slice).groupby("time.month").mean("time")

        x_anom = x_da.groupby("time.month") - x_clim
        obs_mask = xr.where(np.isnan(true_obs_da), False, True)

        obs_mask_on_model_time = obs_mask.reindex(
            # time=x_anom.time,
            time=x_anom.time.values,
            # time=time_slice,
            method=None
        )
        # 稀疏化：只保留“该月真实有观测”的格点
        # x_anom = x_anom.where(obs_mask_on_model_time) #跳过掩膜，因为输入的增广数据已经是稀疏数据
        y_clim = y_da.sel(time=ref_slice).groupby("time.month").mean("time")
        y_anom = y_da.groupby("time.month") - y_clim

        X_train_list.append(x_anom)
        Y_train_list.append(y_anom)
        if f == training_model_files[0]:
            print("[DEBUG] After sparsification:")
            print("  mean finite points per time:",
                  float(np.isfinite(x_anom).sum(("lat", "lon")).mean().values))

    if not X_train_list:
        raise ValueError("No valid X–Y training pairs.")

    print(f"   -> {len(X_train_list)} training samples loaded.")

    # ------------------------------------------------------------------
    # Step 3. Load observation sources
    # ------------------------------------------------------------------
    print("\n--- Loading observation sources ---")

    obs_model_da = _load_and_process_file(
        data_cfg["observation_source_model_path"],
        varname,
        input_depth,
        time_slice,
        is_obs_source=False,
    )
    # 这里输入的是深度段的均值。
    y_path = find_corresponding_y_file(
        data_cfg["observation_source_model_path"],
        y_cfg["source_dir"]
    )
    obs_model_y_da = _load_and_process_target_y(
        y_path, varname, time_slice, y_cfg["depth_range"]
    )

    # Dynamic observation mask
    obs_mask = xr.where(np.isnan(true_obs_da), False, True)

    base_time = obs_mask.time

    print(f"   -> Using mask's time axis as the base for reindexing ({len(base_time)} timesteps).")
    inject_uncertainty = data_cfg.get('inject_uncertainty', False)
    #在这里导入不确定性集合
    bias_ensemble_da = None
    if inject_uncertainty:
        print("--- UNCERTAINTY INJECTION: ENABLED ---")
        bias_path = data_cfg.get('bias_ensemble_path')
        if bias_path and os.path.exists(bias_path):
            print("   -> Loading bias ensemble for uncertainty injection...")
            bias_ensemble_da_raw = _load_and_process_file(bias_path, 'o2_bias', -1, time_slice)  # depth_level=-1表示不选深度
            # 对齐偏差集合的时间轴
            if bias_ensemble_da_raw is not None:
                # 使用我们刚刚定义的基准时间轴来对齐偏差集合
                bias_ensemble_da = bias_ensemble_da_raw.reindex(time=base_time, method='nearest')
                print(f"   -> Bias ensemble loaded and aligned. Shape: {bias_ensemble_da.shape}")
    else:
        print("--- UNCERTAINTY INJECTION: DISABLED ---")

    # ------------------------------------------------------------------
    # Step 4. Observation anomalies
    # ------------------------------------------------------------------
    obs_model_clim = obs_model_da.sel(time=ref_slice).groupby("time.month").mean("time")
    obs_model_anom = obs_model_da.groupby("time.month") - obs_model_clim

    # Model observation Y (global mean anomaly)
    obs_model_y_clim = obs_model_y_da.sel(time=ref_slice).groupby("time.month").mean("time")
    obs_model_y_anom_truth = obs_model_y_da.groupby("time.month") - obs_model_y_clim
    obs_model_y_clim_mean = obs_model_y_clim.mean()
    print(f"   -> Model observation source's Y (0-100m) climatology mean: {float(obs_model_y_clim_mean.compute()):.6f}")

    obs_model_y_anom = calculate_weighted_mean(obs_model_anom)

    # ------------------------------------------------------------------
    # Step 5. True observation anomalies (WOA reference)
    # ------------------------------------------------------------------

    true_obs_anom = None
    true_obs_y_anom = None
    true_obs_climatology_regional_mean = None

    woa_cfg = data_cfg["woa18_climatology"]

    # --- 1. Load WOA monthly climatology ---
    woa_surface_monthly = []
    woa_target_monthly = []

    for m in range(1, 13):
        fpath = glob.glob(
            woa_cfg["path_pattern"].replace("*", f"{m:02d}")
        )[0]

        with xr.open_dataset(fpath, decode_times=False) as ds:
            da = ds[woa_cfg["var_name"]]

            # surface
            woa_surface = da.isel(depth=0).squeeze(drop=True)

            # target depth mean (e.g. 0–100 m)
            woa_target = calculate_vertical_mean(
                da, data_cfg["target_y_config"]["depth_range"]
            ).squeeze(drop=True)

            spatial_coords = {"lat", "lon", "latitude", "longitude"}
            woa_surface = woa_surface.drop_vars(
                [c for c in woa_surface.coords if c not in spatial_coords]
            )
            woa_target = woa_target.drop_vars(
                [c for c in woa_target.coords if c not in spatial_coords]
            )

            woa_surface_monthly.append(woa_surface.load())
            woa_target_monthly.append(woa_target.load())

    woa_surface_clim = xr.concat(woa_surface_monthly, dim="month")
    woa_target_clim = xr.concat(woa_target_monthly, dim="month")

    woa_surface_clim = woa_surface_clim.assign_coords(month=np.arange(1, 13))
    woa_target_clim = woa_target_clim.assign_coords(month=np.arange(1, 13))

    factor = woa_cfg.get("unit_conversion_factor", 1.0)
    woa_surface_clim *= factor
    woa_target_clim *= factor

    print(f"   -> WOA surface climatology shape: {woa_surface_clim.shape}")
    print(f"   -> WOA 0–100m climatology shape: {woa_target_clim.shape}")

    # --- 2. Compute true observation anomalies (surface-based) ---
    true_obs_anom = (
            true_obs_da.groupby("time.month")
            - woa_surface_clim
    )

    # --- 3. Regional mean anomaly for Y ---
    true_obs_y_anom = calculate_weighted_mean(true_obs_anom)

    # --- 4. Reference climatology mean for absolute recovery ---
    true_obs_climatology_regional_mean = calculate_weighted_mean(
        woa_target_clim.mean("month")
    )

    print(
        f"   -> Reference climatology mean for absolute value recovery "
        f"(WOA 0–100 m): "
        f"{float(true_obs_climatology_regional_mean.compute()):.6f}"
    )

    assert true_obs_y_anom is not None

    # ------------------------------------------------------------------
    # Step 6. Align all time axes
    # ------------------------------------------------------------------
    X_train_aligned = [x.reindex(time=base_time, method="nearest") for x in X_train_list]
    Y_train_aligned = [y.reindex(time=base_time, method="nearest") for y in Y_train_list]

    obs_model_anom = obs_model_anom.reindex(time=base_time, method="nearest")
    obs_model_y_anom = obs_model_y_anom.reindex(time=base_time, method="nearest")
    true_obs_anom = true_obs_anom.reindex(time=base_time, method="nearest")
    true_obs_y_anom = true_obs_y_anom.reindex(time=base_time, method="nearest")

    # ------------------------------------------------------------------
    # Step 7. Stack training arrays
    # ------------------------------------------------------------------

    groups_list = [np.full(len(ds.time), i) for i, ds in enumerate(X_train_aligned)]
    training_groups = np.concatenate(groups_list)

    X_train = xr.concat(X_train_aligned, dim="model")
    Y_train = xr.concat(Y_train_aligned, dim="model")

    X_values = (
        X_train
        .stack(sample=("model", "time"))
        .transpose("sample", "lat", "lon")
        .compute()
        .values
    )

    Y_values = (
        Y_train
        .stack(sample=("model", "time"))
        .compute()
        .values
    )

    mask_values = (
        obs_mask
        .broadcast_like(X_train)
        .stack(sample=("model", "time"))
        .transpose("sample", "lat", "lon")
        .compute()
        .values
        > 0
    )

    # ------------------------------------------------------------------
    # Step 8. Normalization stats
    # ------------------------------------------------------------------
    stats_base = xr.concat(X_train_list, dim="sample")
    mean_stat = float(stats_base.mean().compute())
    std_stat = float(stats_base.std().compute())

    # ------------------------------------------------------------------
    # Return dictionary
    # ------------------------------------------------------------------
    return {
        "training_da_values": X_values, #表层异常场 X
        "Y_train_truth_values": Y_values, #所有训练样本对应的真实目标Y异常值 ： 0-100m深度段的DO平均浓度异常值
        "training_mask_da_values": mask_values, #与 X 同形状的观测可用性掩膜
        "training_groups": training_groups,#每个 sample 属于哪一个“模式成员”，主要是为了按模式成员做交叉验证

        "observation_source_model_da_anom_values": obs_model_anom.compute().values, #模式观测源的 表层异常场时间序列
        # "Y_obs_source_truth_anom_values": obs_model_y_anom.compute().values,
        "Y_obs_source_truth_anom_values": obs_model_y_anom_truth.values,  # 模式观测源对应的 目标 Y 的真实异常时间序列
        "bias_ensemble_values": bias_ensemble_da.values if bias_ensemble_da is not None else None,
        "true_obs_climatology_regional_mean": float(true_obs_climatology_regional_mean.compute()), # 真实观测值的平均，其实就是WOA的数据
        "obs_model_climatology_regional_mean": float(obs_model_y_clim_mean.compute()),

        "observation_source_true_da_anom_values": true_obs_anom.compute().values, #真实观测的 表层异常场时间序列
        "obs_mask_da_values": obs_mask.values,  # 或者 .compute().values，这是真实观测可用性掩膜（原始版本）

        "time_coord": base_time.values, #所有数据统一后的时间轴
        "coords": {
            "lat": obs_mask.lat.values,
            "lon": obs_mask.lon.values,
        },
        "stats": {
            "mean": mean_stat,
            "std": std_stat,
        },
        "woa_surface_climatology": woa_surface_clim,
        "woa_target_climatology": woa_target_clim,
    }
