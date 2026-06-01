"""
wod_data_process.py — Process raw WOD (World Ocean Database) oxygen data
into gridded monthly NetCDF files.

Pipeline for each dataset type (OSD, CTD, PFL):
  1. Read raw NetCDF profiles from WOD
  2. Extract surface-layer oxygen measurements with QC flag = 0
  3. Compute oxygen concentration in mol m⁻³
  4. Bin observations to a 1°×1° monthly grid
  5. Save as NetCDF, ready for the reconstruction pipeline

Supports splitting by dataset type and by country.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import netCDF4 as nc
import gsw

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = "/g12338011ghq/dataset/WOD/WOD_raw_data"
OUTPUT_DIR = "/g12338011ghq/dataset/WOD/Subsets_MultiLevel"
TARGET_YEAR_RANGE = (1878, 2017)

DEPTH_LAYERS = [
    {"label": 0,   "min": 0,   "max": 10},
    {"label": 100, "min": 90,  "max": 110},
    {"label": 200, "min": 190, "max": 210},
]

SUBSET_TASKS = {
    "ALL":       lambda df: pd.Series(True, index=df.index),
    "Type_OSD":  lambda df: df["dataset_type"] == "OSD",
    "Type_CTD":  lambda df: df["dataset_type"] == "CTD",
    "Type_PFL":  lambda df: df["dataset_type"] == "PFL",
    "Country_US": lambda df: df["country"] == "UNITED STATES",
    "Country_JP": lambda df: df["country"] == "JAPAN",
    "Country_UK": lambda df: df["country"] == "GREAT BRITAIN",
    "Country_DE": lambda df: df["country"] == "GERMANY",
    "Country_SU": lambda df: df["country"].isin(
        ["SOVIET UNION", "RUSSIAN FEDERATION"]
    ),
}

LON_BINS = np.arange(-180, 181, 1)
LAT_BINS = np.arange(-90, 91, 1)
# ---------------------------------------------------------------------------


def _get_row_size_var(ds):
    """Find the row-size variable in a WOD NetCDF file."""
    for candidate in ("Oxygen_row_size", "z_row_size"):
        if candidate in ds.variables:
            return ds.variables[candidate][:]
    for v in ds.variables:
        if v.endswith("row_size") and "WOD" not in v:
            return ds.variables[v][:]
    return None


def process_single_file(filepath: str, dataset_type: str):
    """
    Read a single WOD raw NetCDF file and return a DataFrame of
    quality-controlled oxygen measurements at the defined depth layers.

    Returns None if no valid data can be extracted.
    """
    try:
        ds = nc.Dataset(filepath)
    except Exception:
        return None

    if "Oxygen" not in ds.variables:
        ds.close()
        return None

    row_size = _get_row_size_var(ds)
    if row_size is None:
        ds.close()
        return None

    # ---- 1. Core variables (must succeed) ----
    try:
        lats = np.repeat(ds.variables["lat"][:], row_size)
        lons = np.repeat(ds.variables["lon"][:], row_size)
        dates = np.repeat(ds.variables["date"][:], row_size)
        depth = ds.variables["z"][:].copy()
        o2 = ds.variables["Oxygen"][:].copy()
        o2_flag = ds.variables["Oxygen_WODflag"][:].copy()
        obs_len = len(o2)
    except Exception:
        ds.close()
        return None

    # ---- 2. Auxiliary variables (best-effort) ----
    try:
        if "country" in ds.variables:
            raw_c = ds.variables["country"][:]
            try:
                country_list = nc.chartostring(raw_c, encoding="utf-8")
                country_list = np.char.strip(country_list)
            except Exception:
                country_list = np.array([
                    b"".join(row).decode("utf-8", errors="ignore").strip()
                    for row in raw_c
                ])
            countries = np.repeat(country_list, row_size)
        else:
            raise ValueError
    except Exception:
        countries = np.full(obs_len, "Unknown")

    try:
        if (
            "Temperature" in ds.variables
            and len(ds.variables["Temperature"]) == obs_len
        ):
            temp = ds.variables["Temperature"][:].copy()
            t_flag = ds.variables["Temperature_WODflag"][:].copy()
        else:
            raise ValueError
    except Exception:
        temp = np.full(obs_len, np.nan)
        t_flag = np.ones(obs_len)

    try:
        if (
            "Salinity" in ds.variables
            and len(ds.variables["Salinity"]) == obs_len
        ):
            salt = ds.variables["Salinity"][:].copy()
            s_flag = ds.variables["Salinity_WODflag"][:].copy()
        else:
            raise ValueError
    except Exception:
        salt = np.full(obs_len, np.nan)
        s_flag = np.ones(obs_len)

    ds.close()

    # ---- 3. Build DataFrame and apply QC ----
    df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "date": dates,
        "country": countries,
        "dataset_type": dataset_type,
        "depth": depth,
        "o2": o2,
        "o2_flag": o2_flag,
        "temp": temp,
        "t_flag": t_flag,
        "salt": salt,
        "s_flag": s_flag,
    })

    # Keep only QC-flagged good data
    df = df[df["o2_flag"] == 0].copy()
    if df.empty:
        return None

    # ---- 4. Extract depth layers ----
    frames = []
    for layer in DEPTH_LAYERS:
        mask = (
            (df["depth"] >= layer["min"])
            & (df["depth"] <= layer["max"])
        )
        sub = df[mask].copy()
        if not sub.empty:
            sub["lev"] = layer["label"]
            frames.append(sub)

    if not frames:
        return None

    df_out = pd.concat(frames, ignore_index=True)

    # Density (default 1025 kg/m³, compute from T/S when possible)
    df_out["density"] = 1025.0
    good_ts = (df_out["t_flag"] == 0) & (df_out["s_flag"] == 0)
    if good_ts.any():
        try:
            sub = df_out[good_ts]
            df_out.loc[good_ts, "density"] = gsw.rho_t_exact(
                sub["salt"], sub["temp"], sub["depth"]
            )
        except Exception:
            pass

    # Convert to mol m⁻³
    df_out["o2_mol_m3"] = df_out["o2"] * df_out["density"] * 1e-6
    df_out["year"] = (df_out["date"] // 10000).astype(int)
    df_out["month"] = ((df_out["date"] % 10000) // 100).astype(int)

    # Filter by year range
    y0, y1 = TARGET_YEAR_RANGE
    df_out = df_out[
        (df_out["year"] >= y0) & (df_out["year"] <= y1)
    ]

    return df_out[
        ["lat", "lon", "year", "month", "country",
         "dataset_type", "lev", "o2_mol_m3"]
    ]


def grid_and_save(df: pd.DataFrame, task_name: str) -> None:
    """
    Bin the DataFrame to a 1°×1° monthly grid and save as NetCDF.
    """
    out_path = os.path.join(OUTPUT_DIR, f"WOD_O2_{task_name}.nc")
    if df.empty:
        print(f"  [skip] {task_name}: empty")
        return

    print(f"  -> {os.path.basename(out_path)} (N={len(df)})")

    df = df.copy()
    df["time"] = pd.to_datetime({
        "year": df["year"], "month": df["month"], "day": 15,
    })
    df["lat_bin"] = pd.cut(
        df["lat"], bins=LAT_BINS, labels=LAT_BINS[:-1] + 0.5
    )
    df["lon_bin"] = pd.cut(
        df["lon"], bins=LON_BINS, labels=LON_BINS[:-1] + 0.5
    )

    grouped = (
        df.groupby(["time", "lev", "lat_bin", "lon_bin"])["o2_mol_m3"]
        .mean()
        .reset_index()
    )
    ds = grouped.set_index(
        ["time", "lev", "lat_bin", "lon_bin"]
    ).to_xarray()
    ds = ds.rename({"lat_bin": "lat", "lon_bin": "lon", "o2_mol_m3": "o2"})

    # Fill missing times
    full_time = pd.date_range(
        start=f"{TARGET_YEAR_RANGE[0]}-01-15",
        end=f"{TARGET_YEAR_RANGE[1]}-12-15",
        freq="MS",
    ) + pd.Timedelta(days=14)
    ds = ds.reindex(time=full_time)
    ds = ds.reindex(lev=[l["label"] for l in DEPTH_LAYERS])

    ds["o2"].attrs = {"units": "mol m-3"}
    ds.to_netcdf(
        out_path,
        encoding={"o2": {"zlib": True, "complevel": 5}},
    )


def main() -> None:
    """Main entry point: process all dataset types and subsets."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_data = []
    sub_dirs = ["OSD", "CTD", "PFL"]

    print("=== Processing WOD data ===")

    for sub in sub_dirs:
        folder = os.path.join(BASE_DIR, sub)
        nc_files = sorted(
            glob.glob(os.path.join(folder, "*.nc"))
        )
        print(f"  {sub}: {len(nc_files)} file(s)")
        for f in nc_files:
            part = process_single_file(f, dataset_type=sub)
            if part is not None:
                all_data.append(part)

    if not all_data:
        print("Error: no valid data found.")
        return

    total_rows = sum(len(x) for x in all_data)
    print(f"Merging ... ({total_rows} rows)")
    full_df = pd.concat(all_data, ignore_index=True)

    print("Grouping and gridding ...")
    for name, filt in SUBSET_TASKS.items():
        subset = full_df[filt(full_df)].copy()
        grid_and_save(subset, name)

    print("Done.")


if __name__ == "__main__":
    main()
