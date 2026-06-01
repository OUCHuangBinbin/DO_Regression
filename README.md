# Global Surface Ocean Deoxygenation Reconstruction (MGSR)

This repository contains the code, reconstructed datasets, and supporting materials for the study:

**Reconstructing Historical Global Surface Ocean Deoxygenation from Sparse Observations Using Model-Guided Statistical Reconstruction (MGSR)**

Submitted to *Global Biogeochemical Cycles* (GBC).

---

## Overview

Long-term observations of dissolved oxygen (DO) are spatially sparse and temporally heterogeneous, making it difficult to quantify historical global oxygen changes using traditional spatial interpolation methods.

This repository implements the **Model-Guided Statistical Reconstruction (MGSR)** framework, which directly reconstructs the **global mean surface dissolved oxygen anomaly** from sparse observations without first reconstructing a complete spatial field.

The approach combines:

* Sparse observational constraints from WOD23
* Physical covariance information from CMIP6 Earth System Models
* Ridge regression with leave-one-model-out cross-validation
* Ensemble-based uncertainty quantification

The resulting reconstruction provides a continuous estimate of global surface ocean oxygen variability and deoxygenation trends from historical observations.

---

## Scientific Motivation

Most existing ocean oxygen reconstructions follow:

```text
Sparse Observations
        ↓
Spatial Field Reconstruction
        ↓
Global Mean Oxygen
```

MGSR instead directly estimates:

```text
Sparse Observations
        ↓
Model-Guided Statistical Reconstruction
        ↓
Global Mean Oxygen
```

This avoids the accumulation of interpolation errors under extremely sparse observational coverage and focuses directly on recovering large-scale temporal variability.

---

## Repository Structure

```text
src/
├── dataset.py
│   Data loading and preprocessing pipeline
│
├── models.py
│   Ridge regression models and leave-one-model-out validation
│
├── utils.py
│   Reconstruction, evaluation, visualization and filtering utilities
│
├── main.py
│   Main reconstruction workflow
│
├── create_bias_ensemble.py
│   Build CMIP6-based bias ensemble
│
├── wod_data_process.py
│   Process raw WOD observations
│
├── remap.py
│   Grid remapping and resolution conversion
│
├── calc_deoxygenation_rate.py
│   Linear trend estimation and uncertainty analysis
│
├── calc_csv_metrics.py
│   Statistical evaluation metrics
│
├── plot_method_comparison.py
│   Comparison among MGSR, MME, OLS and naive approaches
│
└── GODIP-O2.py
    Comparison with external dissolved oxygen products
```

---

## Input Data

### Observations

* World Ocean Database 2023 (WOD23)
* World Ocean Atlas 2023 (WOA23)

### Climate Model Simulations

CMIP6 Earth System Models including:

* ACCESS-ESM1-5
* CanESM5
* CNRM-ESM2-1
* MPI-ESM-1-2-HAM
* MPI-ESM1-2-HR
* MPI-ESM1-2-LR

and associated ensemble members.

---

## Reconstruction Workflow

### Step 1. Observation Processing

```text
Raw WOD Profiles
        ↓
Quality Control
        ↓
Monthly Gridded Fields
        ↓
Global Sparse Observation Matrix
```

### Step 2. Model Training

```text
CMIP6 Fields
        ↓
Global Mean Oxygen
        ↓
Ridge Regression Training
```

### Step 3. Reconstruction

```text
Sparse WOD Observations
        ↓
MGSR Prediction
        ↓
Global Mean Surface DO Reconstruction
```

### Step 4. Uncertainty Quantification

```text
CMIP6 Ensemble Spread
        ↓
Bias Ensemble Generation
        ↓
95% Confidence Interval
```

---

## Requirements

Python ≥ 3.9

Main dependencies:

```bash
numpy
scipy
pandas
xarray
netCDF4
scikit-learn
matplotlib
tqdm
pyarrow
gsw
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Reconstruction

```bash
python src/main.py
```

### Deoxygenation Trend Analysis

```bash
python src/calc_deoxygenation_rate.py
```

### Reconstruction Evaluation

```bash
python src/calc_csv_metrics.py
```

---

## Reconstructed Dataset

The repository contains the reconstructed global mean surface dissolved oxygen dataset used in the manuscript.

Main variables include:

| Variable   | Description                                  |
| ---------- | -------------------------------------------- |
| Year       | Calendar year                                |
| DO_Anomaly | Global mean surface dissolved oxygen anomaly |
| Lower95    | Lower 95% uncertainty bound                  |
| Upper95    | Upper 95% uncertainty bound                  |

---

## Outputs

Running the reconstruction workflow produces:

### Figures

* anomaly_comparison_with_uncertainty.png
* final_absolute_comparison_with_uncertainty.png
* low_pass_filtered_comparison.png
* high_pass_filtered_comparison.png

### Data Products

* reconstruction_results.parquet
* Ensemble_All_Members_Annual_Anom.csv
* annual reconstruction time-series
* monthly reconstruction time-series

---

## Citation

If you use this code or dataset, please cite:

Huang, B., et al. (2026). Reconstructing Historical Global Surface Ocean Deoxygenation from Sparse Observations Using Model-Guided Statistical Reconstruction. *Global Biogeochemical Cycles* (submitted).

---

## License

MIT License

See LICENSE for details.

---

## Contact

Binbin Huang

Email: huangbinbin@zju.edu.cn

GitHub:
https://github.com/OUCHuangBinbin

For questions regarding the reconstruction framework, datasets, or code implementation, please open an issue in this repository.
