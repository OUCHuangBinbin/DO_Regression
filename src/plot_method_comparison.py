"""
plot_method_comparison.py
=========================
Compare reconstruction methods (Ridge, MME, Naive Mean, OLS) against the
CMIP6 truth using pre-computed CSV output.

Produces:
  - Time-series comparison plots
  - Rolling RMSE curves
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from dataset import load_and_prepare_data


def save_to_csv(df: pd.DataFrame, save_path: str) -> None:
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"Results saved to: {save_path}")


def visualize_from_csv(
    csv_path: str,
    plot_trend: bool = True,
    plot_rmse: bool = True,
    methods: list = None,
) -> None:
    """
    Read a CSV and produce comparison plots.

    Args:
        csv_path: Path to the CSV file.
        plot_trend: Whether to plot the time-series comparison panel.
        plot_rmse: Whether to plot the rolling RMSE panel.
        methods: List of method column names to include.
    """
    if methods is None:
        methods = ["Ridge", "MME", "Naive", "OLS"]

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    n_plots = sum([plot_trend, plot_rmse])
    if n_plots == 0:
        return

    fig, axes = plt.subplots(
        n_plots, 1, figsize=(16, 6 * n_plots), dpi=150
    )
    if n_plots == 1:
        axes = [axes]

    style = {
        "Ridge": {"color": "red",   "label": "Ridge"},
        "Naive": {"color": "blue",  "label": "Naive Mean"},
        "MME":   {"color": "cyan",  "label": "MME"},
        "OLS":   {"color": "green", "label": "OLS"},
    }

    ax_idx = 0
    if plot_trend:
        ax = axes[ax_idx]
        if "Truth" in df.columns:
            ax.plot(df.index, df["Truth"], color="black",
                    lw=2.5, label="Truth", zorder=5)
        for m in methods:
            if m in df.columns:
                s = style.get(m, {"color": "gray", "label": m})
                ax.plot(df.index, df[m], color=s["color"],
                        lw=1.5, alpha=0.8, label=s["label"])
        ax.set_title("Global Surface DO Anomaly: Method Comparison",
                     loc="left", fontsize=14, fontweight="bold")
        ax.set_ylabel("Anomaly (mol m-3)")
        ax.legend()
        ax_idx += 1

    if plot_rmse:
        ax = axes[ax_idx]
        for m in methods:
            col = f"RMSE_{m}"
            if col in df.columns:
                s = style.get(m, {"color": "gray", "label": m})
                ax.plot(df.index, df[col], color=s["color"],
                        lw=2, label=f"{s['label']} (RMSE)")
        ax.axhline(0, color="gray", lw=1, alpha=0.5)
        ax.set_title("12-Month Rolling RMSE",
                     loc="left", fontsize=14, fontweight="bold")
        ax.set_ylabel("Error (mol m-3)")
        ax.legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main entry point."""
    config_path = "/g12338011ghq/project/hbb/OxygenDiffusion/" \
                  "Oxy_regression_Surface/config/Final_config.yaml"
    csv_path = "/g12338011ghq/project/hbb/OxygenDiffusion/" \
               "Oxy_regression_Surface/results/Results_CSV/" \
               "WOD_1900_2014_Ship.csv"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading existing results: {os.path.basename(csv_path)}")
    df_csv = pd.read_csv(csv_path, index_col="time", parse_dates=True)

    print("Loading raw data for MME and Naive Mean computation ...")
    config["data"]["inject_uncertainty"] = False
    data = load_and_prepare_data(config)

    time_coord = pd.to_datetime(data["time_coord"])
    n_times = len(time_coord)

    # Compute MME (time-varying multi-model mean)
    groups = data["training_groups"]
    n_models = len(np.unique(groups))
    Y_train = data["Y_train_truth_values"]
    Y_reshaped = Y_train.reshape(n_models, n_times)
    mme_series = np.nanmean(Y_reshaped, axis=0)

    # Compute Naive Mean (area-weighted)
    lats = data["coords"]["lat"]
    lat_weights = np.cos(np.deg2rad(lats))
    weights_2d = np.tile(
        lat_weights[:, np.newaxis], (1, len(data["coords"]["lon"]))
    )
    X_source_anom = data["observation_source_model_da_anom_values"]
    obs_mask = data["obs_mask_da_values"]

    naive_series = []
    for t in range(n_times):
        field = X_source_anom[t]
        mask_t = obs_mask[t] > 0
        obs_vals = field[mask_t]
        obs_w = weights_2d[mask_t]
        valid = ~np.isnan(obs_vals)
        if np.any(valid):
            naive_series.append(
                np.average(obs_vals[valid], weights=obs_w[valid])
            )
        else:
            naive_series.append(np.nan)
    naive_series = np.array(naive_series)

    # Build comparison DataFrame
    df_final = pd.DataFrame(index=time_coord)
    df_final["Truth"] = df_csv["anom_cmip6_truth"]
    df_final["Ridge"] = df_csv["anom_recon_model_src"]
    df_final["OLS"] = df_csv["OLS_Recon"]
    df_final["MME"] = mme_series
    df_final["Naive"] = naive_series

    methods = ["Ridge", "MME", "Naive", "OLS"]
    for m in methods:
        error = df_final[m] - df_final["Truth"]
        df_final[f"RMSE_{m}"] = np.sqrt(
            (error**2).rolling(window=12, center=True).mean()
        )
        df_final[f"Bias_{m}"] = error.rolling(
            window=12, center=True
        ).mean()

    # Save
    final_csv = os.path.join(
        os.path.dirname(csv_path),
        "Final_Comparison_Data.csv",
    )
    save_to_csv(df_final, final_csv)

    # Plot
    visualize_from_csv(
        csv_path,
        plot_trend=True,
        plot_rmse=True,
        methods=["Ridge", "MME", "Naive", "OLS"],
    )


if __name__ == "__main__":
    main()
