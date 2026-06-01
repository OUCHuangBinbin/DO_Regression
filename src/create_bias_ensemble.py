"""
create_bias_ensemble.py — Create a bias ensemble for uncertainty injection.

The bias ensemble quantifies structural uncertainty in the CMIP6
multi-model ensemble.  For each member, the bias is defined as its
deviation from the multi-model mean (MMM):
    bias_i = model_i − MMM

The resulting ensemble has shape (n_members, time, lat, lon) and is
saved to a single NetCDF file for later use in uncertainty propagation.
"""

import os
import glob
import yaml
import numpy as np
import xarray as xr
from dataset import _load_and_process_file


def main() -> None:
    # ------------------------------------------------------------------
    # 1.  Load configuration
    # ------------------------------------------------------------------
    config_path = "/g12338011ghq/project/hbb/OxygenDiffusion/" \
                  "Oxy_regression/config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    output_path = data_cfg.get("bias_ensemble_path")
    if not output_path:
        print("[ERROR] 'bias_ensemble_path' not defined in config.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ------------------------------------------------------------------
    # 2.  Discover and load all training model files
    # ------------------------------------------------------------------
    print("\n--- [Step 1] Loading training models ---")
    training_dirs = data_cfg.get("training_model_dirs", [])
    all_files = []
    for d in training_dirs:
        all_files.extend(
            glob.glob(os.path.join(d, "**", "*.nc"), recursive=True)
        )
    all_files = sorted(set(all_files))

    # Exclude observation files and the bias ensemble file itself
    exclude = {
        os.path.abspath(p) for p in [
            data_cfg.get("observation_source_model_path"),
            data_cfg.get("obs_mask_path"),
            data_cfg.get("observation_source_true_path"),
            data_cfg.get("bias_ensemble_path"),
        ] if p
    }
    model_files = [
        f for f in all_files
        if os.path.abspath(f) not in exclude
    ]

    if len(model_files) < 2:
        print("[ERROR] Need at least 2 training models. Exiting.")
        return

    print(f"Found {len(model_files)} model(s).")

    time_slice = slice(
        f"{data_cfg['time_range'][0]}-01-01",
        f"{data_cfg['time_range'][-1]}-12-31",
    )

    model_data = [
        _load_and_process_file(
            f, data_cfg.get("variable_name", "o2"),
            data_cfg["depth_level"], time_slice,
            is_obs_source=False,
        )
        for f in model_files
    ]
    model_data = [ds for ds in model_data if ds is not None]

    if not model_data:
        print("[ERROR] No models loaded. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3.  Align all models to a common time axis via reindex
    # ------------------------------------------------------------------
    print("\n--- [Step 2] Aligning time axes ---")
    base_time = model_data[0].time
    print(f"     -> Reference time axis: {len(base_time)} steps.")

    aligned = [model_data[0]]
    for ds in model_data[1:]:
        aligned.append(ds.reindex(time=base_time, method="nearest"))

    # ------------------------------------------------------------------
    # 4.  Compute bias ensemble (model − MMM)
    # ------------------------------------------------------------------
    print("\n--- [Step 3] Computing bias ensemble ---")
    combined = xr.concat(aligned, dim="member")
    mmm = combined.mean(dim="member")

    bias_ensemble = combined - mmm
    bias_ensemble.name = "o2_bias"

    print(f"     -> Bias ensemble shape: {bias_ensemble.shape}")

    # ------------------------------------------------------------------
    # 5.  Save
    # ------------------------------------------------------------------
    print(f"\n--- [Step 4] Saving to {output_path} ---")
    encoding = {
        "o2_bias": {
            "dtype": "float32",
            "_FillValue": -9999.0,
            "zlib": True,
            "complevel": 4,
        }
    }
    bias_ensemble.to_netcdf(output_path, encoding=encoding)
    print("Done.")


if __name__ == "__main__":
    main()
