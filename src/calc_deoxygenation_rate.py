"""
calc_deoxygenation_rate.py
==========================
Compute surface-ocean deoxygenation rates from annual mean DO time-series.

Units: μmol m⁻³ decade⁻¹
Method: ordinary least-squares (OLS) linear regression with Bootstrap
        resampling (n=2000) to estimate 95 % confidence intervals.

All parameters are set in main() at the bottom of this file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Optional, Dict


# ======================================================================
# Section 1 — Data I/O
# ======================================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV indexed by DatetimeIndex."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"[load]  {os.path.basename(csv_path)}  ->  "
          f"{df.shape[0]} rows x {df.shape[1]} cols")
    return df


def to_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Resample monthly DataFrame to annual means (index = integer year)."""
    annual = df.resample("YE").mean()
    annual.index = annual.index.year
    annual.index.name = "year"
    return annual


# ======================================================================
# Section 2 — Unit conversion
# ======================================================================

MOL_TO_UMOL = 1e6          # mol → μmol
YEARS_PER_DEC = 10.0       # 1 decade


def slope_to_rate(slope_mol_per_yr: float) -> float:
    """Convert linear slope (mol m⁻³ yr⁻¹) to μmol m⁻³ decade⁻¹."""
    return slope_mol_per_yr * MOL_TO_UMOL * YEARS_PER_DEC


# ======================================================================
# Section 3 — Trend estimation (OLS + Bootstrap CI)
# ======================================================================

def ols_trend(
    years: np.ndarray,
    values: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Ordinary least-squares linear regression.

    Returns (slope, intercept, r_value, p_value).
    """
    slope, intercept, r, p, _ = stats.linregress(years, values)
    return slope, intercept, r, p


def bootstrap_slope_ci(
    years: np.ndarray,
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for the OLS slope.

    Returns (lo, hi) in mol m⁻³ yr⁻¹.
    """
    rng = np.random.default_rng(seed)
    n = len(years)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s, *_ = stats.linregress(years[idx], values[idx])
        slopes.append(s)
    alpha = (100.0 - ci) / 2.0
    return (
        float(np.percentile(slopes, alpha)),
        float(np.percentile(slopes, 100.0 - alpha)),
    )


def compute_rate(
    annual_df: pd.DataFrame,
    col: str,
    period: Tuple[int, int],
    n_boot: int = 2000,
    min_years: int = 5,
) -> Optional[Dict]:
    """
    Compute deoxygenation rate for a given column and time period.

    Returns None if fewer than min_years valid data points exist.
    """
    y0, y1 = period
    sub = annual_df.loc[y0:y1, col].dropna()

    if len(sub) < min_years:
        print(f"  [skip] {col} {y0}–{y1}: only {len(sub)} year(s)")
        return None

    years = sub.index.values.astype(float)
    values = sub.values

    slope, intercept, r, p = ols_trend(years, values)
    lo, hi = bootstrap_slope_ci(years, values, n_boot=n_boot)

    rate = slope_to_rate(slope)
    rate_lo = slope_to_rate(lo)
    rate_hi = slope_to_rate(hi)

    return {
        "period":          f"{y0}–{y1}",
        "y0":              y0,
        "y1":              y1,
        "n_years":         len(sub),
        "col":             col,
        "slope_mol_yr":    slope,
        "rate_umol_dec":   rate,
        "rate_lo":         rate_lo,
        "rate_hi":         rate_hi,
        "r":               r,
        "p":               p,
        "fit_x":           years,
        "fit_y":           intercept + slope * years,
        "raw_x":           years,
        "raw_y":           values,
    }


# ======================================================================
# Section 4 — Batch computation
# ======================================================================

def compute_all_rates(
    annual_df: pd.DataFrame,
    col: str,
    periods: List[Tuple[int, int]],
    n_boot: int = 2000,
    min_years: int = 5,
) -> pd.DataFrame:
    """Compute rates for multiple periods on a single column."""
    records = []
    for period in periods:
        res = compute_rate(annual_df, col, period, n_boot, min_years)
        if res is not None:
            records.append({
                "period":               res["period"],
                "n_years":              res["n_years"],
                "rate (μmol/m³/dec)":   round(res["rate_umol_dec"], 4),
                "CI_lo":                round(res["rate_lo"], 4),
                "CI_hi":                round(res["rate_hi"], 4),
                "r":                    round(res["r"], 3),
                "p":                    round(res["p"], 4),
                "significant":          res["p"] < 0.05,
            })
    return pd.DataFrame(records)


def compute_rates_multi_cols(
    annual_df: pd.DataFrame,
    cols: List[str],
    periods: List[Tuple[int, int]],
    col_labels: Optional[Dict[str, str]] = None,
    n_boot: int = 2000,
    min_years: int = 5,
) -> pd.DataFrame:
    """
    Compute deoxygenation rates for multiple columns (products) and
    periods, returning a long-format DataFrame.

    Column labels are auto-generated by stripping 'anom_ref_' / 'anom_'
    prefixes unless overridden via col_labels.
    """
    if col_labels is None:
        col_labels = {}

    def auto_label(c: str) -> str:
        for prefix in ("anom_ref_", "abs_ref_", "anom_", "abs_"):
            c = c.replace(prefix, "")
        return c.replace("_", " ").strip()

    all_records = []
    for col in cols:
        if col not in annual_df.columns:
            print(f"[warn]  '{col}' not found, skipping.")
            continue

        label = col_labels.get(col, auto_label(col))
        valid_idx = annual_df[col].dropna().index
        if len(valid_idx) == 0:
            print(f"[warn]  '{col}' has no valid data.")
            continue

        data_start = int(valid_idx.min())
        data_end = int(valid_idx.max())

        for period in periods:
            y0, y1 = period
            if y0 > data_end or y1 < data_start:
                print(f"  [skip] {label} {y0}–{y1}: no overlap "
                      f"({data_start}–{data_end})")
                continue

            y0_eff, y1_eff = max(y0, data_start), min(y1, data_end)
            sub = annual_df.loc[y0_eff:y1_eff, col].dropna()

            if len(sub) < min_years:
                print(f"  [skip] {label} {y0_eff}–{y1_eff}: "
                      f"only {len(sub)} year(s)")
                continue

            years = sub.index.values.astype(float)
            values = sub.values
            slope, _, r, p = ols_trend(years, values)
            lo, hi = bootstrap_slope_ci(years, values, n_boot=n_boot)

            all_records.append({
                "dataset":             label,
                "col":                 col,
                "period":              f"{y0}–{y1}",
                "n_years":             len(sub),
                "rate (μmol/m³/dec)":  round(slope_to_rate(slope), 4),
                "CI_lo":               round(slope_to_rate(lo), 4),
                "CI_hi":               round(slope_to_rate(hi), 4),
                "r":                   round(r, 3),
                "p":                   round(p, 4),
                "significant":         p < 0.05,
            })

    return pd.DataFrame(all_records)


# ======================================================================
# Section 5 — Colour palette (Paul Tol, colourblind-friendly)
# ======================================================================

_COLORS = [
    "#EE6677", "#4477AA", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB",
    "#EE7733", "#0077BB", "#33BBEE", "#009988", "#CC3311",
]


# ======================================================================
# Section 6 — Visualisation
# ======================================================================

def _apply_style(
    ax, xlabel: str = "Year", ylabel: str = ""
) -> None:
    """Journal-style axis formatting."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(axis="both", direction="in", length=3,
                   width=0.6, labelsize=6)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4,
                  color="#AAAAAA", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=7, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def plot_trend_panels(
    annual_df: pd.DataFrame,
    col: str,
    periods: List[Tuple[int, int]],
    n_boot: int = 2000,
    min_years: int = 5,
    title: str = "",
    out: str = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Sub-panels showing annual series + OLS trend for each period.

    Sub-plots are arranged in up to 3 columns.
    """
    results = [
        compute_rate(annual_df, col, p, n_boot, min_years)
        for p in periods
    ]
    results = [r for r in results if r is not None]
    n = len(results)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows),
        squeeze=False,
    )

    for idx, res in enumerate(results):
        ax = axes[idx // ncols][idx % ncols]
        c = _COLORS[idx % len(_COLORS)]

        ax.plot(res["raw_x"], res["raw_y"] * MOL_TO_UMOL,
                color=c, lw=0.8, alpha=0.7, zorder=2)
        ax.plot(res["fit_x"], res["fit_y"] * MOL_TO_UMOL,
                color="#222222", lw=1.2, ls="--", zorder=3,
                label="OLS trend")

        rate = res["rate_umol_dec"]
        lo, hi = res["rate_lo"], res["rate_hi"]
        p_val = res["p"]
        sig = ("**" if p_val < 0.01
               else ("*" if p_val < 0.05 else "n.s."))
        annot = (f"{rate:+.2f} μmol m⁻³ dec⁻¹\n"
                 f"[{lo:+.2f}, {hi:+.2f}]\n"
                 f"r = {res['r']:.2f}, p = {p_val:.3f} {sig}")
        ax.text(0.03, 0.97, annot, transform=ax.transAxes,
                fontsize=5.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CCCCCC", lw=0.4, alpha=0.85))

        _apply_style(ax, xlabel="Year",
                     ylabel="DO anomaly (μmol m⁻³)")
        ax.set_title(res["period"], fontsize=7, pad=3, loc="left")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=8, y=1.01, fontweight="bold")
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save]  {out}")
    else:
        plt.show()
    return fig


def plot_rate_summary(
    rate_tables: Dict[str, pd.DataFrame],
    out: str = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Grouped bar chart comparing deoxygenation rates across datasets,
    with 95 % Bootstrap CI error bars.
    """
    periods_all = []
    for df_r in rate_tables.values():
        periods_all.extend(df_r["period"].tolist())
    periods = list(dict.fromkeys(periods_all))

    n_ds = len(rate_tables)
    n_per = len(periods)
    width = 0.7 / n_ds
    x = np.arange(n_per)

    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * n_per), 3.5))

    for idx, (ds_name, df_r) in enumerate(rate_tables.items()):
        offset = (idx - (n_ds - 1) / 2.0) * width
        rates, lo_errs, hi_errs = [], [], []

        for per in periods:
            row = df_r[df_r["period"] == per]
            if row.empty:
                rates.append(np.nan)
                lo_errs.append(0)
                hi_errs.append(0)
            else:
                r = float(row["rate (μmol/m³/dec)"].iloc[0])
                lo = float(row["CI_lo"].iloc[0])
                hi = float(row["CI_hi"].iloc[0])
                rates.append(r)
                lo_errs.append(r - lo)
                hi_errs.append(hi - r)

        rates = np.array(rates, dtype=float)
        lo_errs = np.array(lo_errs, dtype=float)
        hi_errs = np.array(hi_errs, dtype=float)
        valid = ~np.isnan(rates)

        ax.bar(x[valid] + offset, rates[valid],
               width=width * 0.85,
               color=_COLORS[idx % len(_COLORS)],
               alpha=0.8, label=ds_name, zorder=2)
        ax.errorbar(x[valid] + offset, rates[valid],
                     yerr=[lo_errs[valid], hi_errs[valid]],
                     fmt="none", color="#333333",
                     lw=0.8, capsize=2.5, capthick=0.8, zorder=3)

    ax.axhline(0, color="#555555", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=6, rotation=20, ha="right")
    _apply_style(ax, xlabel="Period",
                 ylabel="Deoxygenation rate\n(μmol m⁻³ decade⁻¹)")
    ax.legend(fontsize=6, frameon=True, framealpha=0.85,
              edgecolor="#CCCCCC", borderpad=0.4)
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save]  {out}")
    else:
        plt.show()
    return fig


def plot_rate_multi_cols(
    df_rates: pd.DataFrame,
    periods_to_plot: Optional[List[str]] = None,
    title: str = "",
    out: str = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Multi-panel bar chart: each panel shows rates for one period,
    coloured by dataset.
    """
    all_periods = df_rates["period"].unique().tolist()
    if periods_to_plot:
        all_periods = [p for p in all_periods if p in periods_to_plot]

    datasets = df_rates["dataset"].unique().tolist()
    color_map = {
        ds: _COLORS[i % len(_COLORS)]
        for i, ds in enumerate(datasets)
    }

    ncols = 2 if len(all_periods) == 4 else min(len(all_periods), 3)
    nrows = (len(all_periods) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
        squeeze=False,
    )

    for pi, period in enumerate(all_periods):
        ax = axes[pi // ncols][pi % ncols]
        sub = df_rates[df_rates["period"] == period].copy()

        x = np.arange(len(sub))
        rates = sub["rate (μmol/m³/dec)"].values
        lo_err = rates - sub["CI_lo"].values
        hi_err = sub["CI_hi"].values - rates
        colors = [color_map[ds] for ds in sub["dataset"]]
        sigs = sub["significant"].values

        ax.bar(x, rates, color=colors, alpha=0.82,
               width=0.6, zorder=2)
        ax.errorbar(x, rates, yerr=[lo_err, hi_err],
                     fmt="none", color="#333333",
                     lw=0.8, capsize=2.5, capthick=0.8, zorder=3)

        for xi, (r, sig) in enumerate(zip(rates, sigs)):
            if sig:
                y_pos = r + (hi_err[xi] if r >= 0
                             else -lo_err[xi]) * 1.1
                ax.text(xi, y_pos, "*", ha="center", va="bottom",
                        fontsize=7, color="#333333")

        ax.axhline(0, color="#555555", lw=0.7, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["dataset"].values,
                           fontsize=5, rotation=35, ha="right")
        _apply_style(ax, xlabel="",
                     ylabel="Rate (μmol m⁻³ dec⁻¹)")
        ax.set_title(period, fontsize=7, pad=3, loc="left")

    for idx in range(len(all_periods), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=8, y=1.01, fontweight="bold")
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save]  {out}")
    else:
        plt.show()
    return fig


# ======================================================================
# main() — modify paths and periods here
# ======================================================================

def main() -> None:
    # Paths
    DIR = "/g12338011ghq/project/hbb/OxygenDiffusion/" \
          "Oxy_regression_Surface/results/Results_CSV"
    CSV_WOD = os.path.join(DIR, "WOD_1900_2014_Ship.csv")
    CSV_IAP = os.path.join(DIR, "IAP_1940_2014_Ship.csv")

    df_wod = load_csv(CSV_WOD)
    df_iap = load_csv(CSV_IAP)
    ann_wod = to_annual(df_wod)
    ann_iap = to_annual(df_iap)

    # Periods
    PERIODS = [
        (1950, 2014), (1960, 2014), (1970, 2014),
        (1980, 2014), (1990, 2014), (2000, 2014),
    ]
    PERIODS_REF = [
        (1970, 2014), (1980, 2014), (1990, 2014), (2000, 2014),
    ]

    # Part A — WOD vs IAP main reconstruction
    COL = "anom_recon_median"
    COL_LABELS = {
        COL: "Ours",
        "anom_ref_GT_ML_RF_ShipArgo_corrected": "GT-ML",
        "anom_ref_GT-OI": "GT-OI",
        "anom_ref_UW": "UW",
        "anom_ref_UTAS": "UTAS",
        "anom_ref_SJTU-JW": "SJTU-JW",
        "anom_ref_SJTU-HZ": "SJTU-HZ",
        "anom_ref_MMM": "MMM",
    }

    print("\n" + "=" * 60)
    print("WOD reconstruction")
    print("=" * 60)
    rates_wod = compute_all_rates(ann_wod, COL, PERIODS)
    print(rates_wod.to_string(index=False))

    print("\n" + "=" * 60)
    print("IAP reconstruction")
    print("=" * 60)
    rates_iap = compute_all_rates(ann_iap, COL, PERIODS)
    print(rates_iap.to_string(index=False))

    # Part B — All reference products
    ref_cols = [c for c in ann_wod.columns if c.startswith("anom_ref_")]
    all_cols = [COL] + ref_cols

    print("\n" + "=" * 60)
    print("All products — deoxygenation rates")
    print("=" * 60)
    rates_all = compute_rates_multi_cols(
        ann_wod, all_cols, PERIODS_REF,
        col_labels=COL_LABELS,
    )
    print(rates_all.to_string(index=False))

    # Save tables
    rates_wod["dataset"] = "WOD"
    rates_iap["dataset"] = "IAP"
    pd.concat([rates_wod, rates_iap], ignore_index=True).to_csv(
        os.path.join(DIR, "deoxygenation_rates_main.csv"), index=False
    )
    rates_all.to_csv(
        os.path.join(DIR, "deoxygenation_rates_all_products.csv"),
        index=False,
    )

    # Figures
    plot_rate_summary({"WOD": rates_wod, "IAP": rates_iap}, dpi=300)
    plot_rate_multi_cols(rates_all, dpi=300)


if __name__ == "__main__":
    main()
