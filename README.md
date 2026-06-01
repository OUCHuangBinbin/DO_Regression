# Global Surface Ocean Deoxygenation Reconstruction

This repository contains the code for reconstructing global mean surface
dissolved oxygen (DO) concentrations from sparse observations using
ridge regression with leave-one-group-out cross-validation.

The method follows the framework of Sippel et al. (2019, Nature
Communications) and is applied to CMIP6 model output and WOD
observations to produce a continuous global mean surface DO time-series
with quantified uncertainty.

## Repository Structure

```
src/
├── dataset.py                  — Data loading and preprocessing pipeline
├── models.py                   — Ridge regression model with LOMO-CV
├── utils.py                    — Shared utilities (reconstruction,
│                                 evaluation, visualisation, filtering)
├── main.py                     — Main training + reconstruction pipeline
│
├── create_bias_ensemble.py     — Build bias ensemble from CMIP6 models
├── wod_data_process.py         — Process raw WOD profiles to gridded
│                                 monthly NetCDF
├── remap.py                    — Resample NetCDF files to coarser grid
│
├── calc_deoxygenation_rate.py  — Compute linear deoxygenation trends
│                                 with Bootstrap CI
├── calc_csv_metrics.py         — Accuracy metrics (RMSE, R², bias, ...)
│                                 between any two time-series
├── plot_method_comparison.py   — Compare Ridge, MME, Naive, OLS methods
└── GODIP-O2.py                 — Compare external DO products
```

## Pipeline Overview

1. **Data preprocessing**
   - `wod_data_process.py`: raw WOD profiles → gridded monthly fields
   - `create_bias_ensemble.py`: CMIP6 models → multi-model bias ensemble
   - `remap.py`: optional downscaling of files to a coarser resolution

2. **Reconstruction** (`main.py`)
   - Loads CMIP6 model fields + WOD observations (`dataset.py`)
   - Computes monthly anomalies relative to a reference period
   - Trains a **clean** ridge model (for the best estimate)
   - Trains a **robust** ridge model with uncertainty injection
   - Reconstructs the global mean DO time-series
   - Produces visualisations and saves results to Parquet / CSV

3. **Analysis**
   - `calc_deoxygenation_rate.py`: deoxygenation trends
   - `calc_csv_metrics.py`: accuracy metrics against reference products
   - `plot_method_comparison.py`: cross-method comparison

## Requirements

See `requirements.txt`. Key dependencies:

- Python ≥ 3.9
- numpy, scipy, pandas, xarray, netCDF4
- scikit-learn
- matplotlib
- tqdm
- pyarrow (for Parquet I/O)
- gsw (for WOD processing)

## Usage

```bash
# Train and reconstruct
python src/main.py

# Compute deoxygenation trends
python src/calc_deoxygenation_rate.py

# Compute accuracy metrics
python src/calc_csv_metrics.py
```

Paths to input data are configured via YAML files. Edit the
`config_path` variable in `main.py` to point to your configuration.

## Outputs

Running `main.py` produces:

- **Figures** (PNG, 300 dpi):
  - `anomaly_comparison_with_uncertainty.png`
  - `final_absolute_comparison_with_uncertainty.png`
  - `low_pass_filtered_comparison.png`
  - `high_pass_filtered_comparison.png`

- **Data**:
  - `reconstruction_results.parquet` (+ YAML metadata)
  - `*_monthly.csv` — monthly time-series
  - `*_annual.csv` — annual mean time-series
  - `Ensemble_All_Members_Annual_Anom.csv` — per-member annual values

## Citation



## License

To be determined.
