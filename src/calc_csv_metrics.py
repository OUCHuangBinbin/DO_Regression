"""
calc_csv_metrics.py
===================
Compute accuracy metrics between any two columns in reconstruction CSV
output files.

Metrics: RMSE, MAE, Pearson r, R², Mean Bias, Std Error

Supports:
  - Single pair comparison
  - One-vs-many batch comparison
  - Pairwise (all-vs-all) comparison
  - Monthly or annual aggregation
  - Scatter plots and time-series + difference plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    r2_score
from itertools import combinations
from typing import List, Optional, Tuple, Dict


# ======================================================================
# Section 1 — Data I/O
# ======================================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV with DatetimeIndex."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"[load] {os.path.basename(csv_path)}  shape={df.shape}")
    return df


def print_columns(df: pd.DataFrame) -> None:
    """Pretty-print grouped column names."""
    groups = {"anom_*": [], "abs_*": [], "other": []}
    for c in df.columns:
        if c.startswith("anom_"):
            groups["anom_*"].append(c)
        elif c.startswith("abs_"):
            groups["abs_*"].append(c)
        else:
            groups["other"].append(c)
    print("\n── Available columns ──────────────────────────────")
    for sec, cols in groups.items():
        if cols:
            print(f"  [{sec}]")
            for c in cols:
                print(f"    {c}")
    print("───────────────────────────────────────────────────\n")


def to_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to annual means with integer year index."""
    annual = df.resample("YE").mean()
    annual.index = annual.index.year
    annual.index.name = "year"
    return annual


# ======================================================================
# Section 2 — Metrics
# ======================================================================

def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    label_true: str = "Reference",
    label_pred: str = "Prediction",
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> Dict:
    """
    Compute accuracy metrics between two aligned series.

    Returns dict with keys: ref, pred, n, RMSE, MAE, R2, r, p,
    MeanBias, StdErr. Returns empty dict if fewer than 3 valid pairs.
    """
    a, b = y_true.align(y_pred, join="inner")

    if year_start is not None:
        a = a.loc[a.index >= year_start]
        b = b.loc[b.index >= year_start]
    if year_end is not None:
        a = a.loc[a.index <= year_end]
        b = b.loc[b.index <= year_end]

    valid = a.notna() & b.notna()
    a, b = a[valid].values, b[valid].values

    if len(a) < 3:
        print(f"  [warn] Only {len(a)} valid pair(s).")
        return {}

    rmse = float(np.sqrt(mean_squared_error(a, b)))
    mae = float(mean_absolute_error(a, b))
    r2 = float(r2_score(a, b))
    corr, p = stats.pearsonr(a, b)
    bias = float(np.mean(b - a))
    std_err = float(np.std(b - a))

    return {
        "ref":      label_true,
        "pred":     label_pred,
        "n":        len(a),
        "RMSE":     round(rmse,   8),
        "MAE":      round(mae,    8),
        "R2":       round(r2,     4),
        "r":        round(corr,   4),
        "p":        round(p,      4),
        "MeanBias": round(bias,   8),
        "StdErr":   round(std_err, 8),
    }


def compute_metrics_batch(
    df: pd.DataFrame,
    ref_col: str,
    compare_cols: List[str],
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    resample: str = "annual",
) -> pd.DataFrame:
    """
    Compare ref_col against each column in compare_cols.

    resample = "annual" | "monthly".
    """
    df_use = to_annual(df) if resample == "annual" else df.copy()
    if ref_col not in df_use.columns:
        raise KeyError(f"ref_col '{ref_col}' not found.")

    records = []
    for col in compare_cols:
        if col not in df_use.columns or col == ref_col:
            continue
        m = compute_metrics(
            df_use[ref_col], df_use[col],
            label_true=ref_col, label_pred=col,
            year_start=year_start, year_end=year_end,
        )
        if m:
            records.append(m)
    return pd.DataFrame(records)


def compute_metrics_pairwise(
    df: pd.DataFrame,
    cols: List[str],
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    resample: str = "annual",
) -> pd.DataFrame:
    """Compute metrics for all column pairs (upper triangle)."""
    df_use = to_annual(df) if resample == "annual" else df.copy()
    records = []
    for c1, c2 in combinations(cols, 2):
        if c1 not in df_use.columns or c2 not in df_use.columns:
            continue
        m = compute_metrics(
            df_use[c1], df_use[c2],
            label_true=c1, label_pred=c2,
            year_start=year_start, year_end=year_end,
        )
        if m:
            records.append(m)
    return pd.DataFrame(records)


def print_metrics_table(df_metrics: pd.DataFrame) -> None:
    """Pretty-print a metrics DataFrame."""
    if df_metrics.empty:
        print("No results.")
        return
    print(f"\n{'─' * 110}")
    print(f"{'Reference':<35} {'Prediction':<35} {'n':>5} "
          f"{'RMSE':>12} {'MAE':>12} {'R²':>8} {'r':>8} {'p':>8} "
          f"{'Bias':>12}")
    print(f"{'─' * 110}")
    for _, row in df_metrics.iterrows():
        sig = ("**" if row["p"] < 0.01
               else ("*" if row["p"] < 0.05 else "n.s."))
        print(
            f"{str(row['ref'])[:34]:<35} "
            f"{str(row['pred'])[:34]:<35} "
            f"{int(row['n']):>5} {row['RMSE']:>12.6f} "
            f"{row['MAE']:>12.6f} {row['R2']:>8.4f} "
            f"{row['r']:>8.4f} {row['p']:>7.4f}{sig} "
            f"{row['MeanBias']:>12.6f}"
        )
    print(f"{'─' * 110}\n")


# ======================================================================
# Section 3 — Colour palette
# ======================================================================

_COLORS = [
    "#EE6677", "#4477AA", "#228833", "#CCBB44",
    "#AA3377", "#66CCEE", "#BBBBBB", "#EE8866",
]


# ======================================================================
# Section 4 — Visualisation
# ======================================================================

def plot_scatter_comparison(
    df: pd.DataFrame,
    ref_col: str,
    compare_cols: List[str],
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    resample: str = "annual",
    out: str = None,
    dpi: int = 300,
) -> Optional[plt.Figure]:
    """
    Scatter plot: each compare_col vs ref_col with R² and RMSE labels.
    """
    df_use = to_annual(df) if resample == "annual" else df.copy()
    valid_cols = [
        c for c in compare_cols
        if c in df_use.columns and c != ref_col
    ]
    n = len(valid_cols)
    if n == 0:
        return None

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for idx, col in enumerate(valid_cols):
        ax = axes[idx // ncols][idx % ncols]
        ref = df_use[ref_col]
        pred = df_use[col]
        a, b = ref.align(pred, join="inner")

        if year_start:
            a, b = a.loc[a.index >= year_start], b.loc[b.index >= year_start]
        if year_end:
            a, b = a.loc[a.index <= year_end], b.loc[b.index <= year_end]
        valid = a.notna() & b.notna()
        a, b = a[valid].values, b[valid].values

        if len(a) < 3:
            ax.set_visible(False)
            continue

        c = _COLORS[idx % len(_COLORS)]
        ax.scatter(a, b, s=16, color=c, alpha=0.65,
                   edgecolors="white", linewidths=0.3, zorder=2)

        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], color="#AAAAAA",
                lw=0.8, ls="--", zorder=1)

        r2 = r2_score(a, b)
        rmse = np.sqrt(mean_squared_error(a, b))
        ax.text(0.05, 0.95,
                f"R² = {r2:.4f}\nRMSE = {rmse:.2e}",
                transform=ax.transAxes, fontsize=6.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CCCCCC", lw=0.4, alpha=0.85))

        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(axis="both", direction="in", length=3,
                       width=0.6, labelsize=6)
        ax.set_xlabel(ref_col.split("_")[-1], fontsize=7)
        ax.set_ylabel(col.split("_")[-1], fontsize=7)
        ax.set_title(col.replace("anom_", "").replace("abs_", "")[:40],
                     fontsize=7, pad=3, loc="left")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Scatter comparison vs. {ref_col}",
                 fontsize=9, y=1.01, fontweight="bold")
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


def plot_timeseries_comparison(
    df: pd.DataFrame,
    ref_col: str,
    compare_cols: List[str],
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    resample: str = "annual",
    out: str = None,
    dpi: int = 300,
) -> Optional[plt.Figure]:
    """
    Two-panel plot: time-series (top) + difference (bottom).
    """
    df_use = to_annual(df) if resample == "annual" else df.copy()
    if year_start:
        df_use = df_use.loc[df_use.index >= year_start]
    if year_end:
        df_use = df_use.loc[df_use.index <= year_end]

    valid_cols = [
        c for c in compare_cols
        if c in df_use.columns and c != ref_col
    ]
    if not valid_cols:
        return None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1]},
    )

    ref = df_use[ref_col]
    ax1.plot(ref.index.values, ref.values, color="#222222",
             lw=1.8, zorder=5,
             label=ref_col.replace("anom_", "").replace("abs_", "")[:30])

    for idx, col in enumerate(valid_cols):
        pred = df_use[col]
        c = _COLORS[idx % len(_COLORS)]
        short = col.replace("anom_", "").replace("abs_", "")[:30]

        a, b = ref.align(pred, join="inner")
        valid = a.notna() & b.notna()

        ax1.plot(a.index.values[valid.values], b.values[valid.values],
                 color=c, lw=1.2, alpha=0.85, label=short)
        diff = b - a
        ax2.plot(diff.index.values[valid.values], diff.values[valid.values],
                 color=c, lw=1.0, alpha=0.8)
        ax2.fill_between(diff.index.values[valid.values],
                         0, diff.values[valid.values],
                         color=c, alpha=0.15)

    ax2.axhline(0, color="#555555", lw=0.7)

    for ax, ttl in [
        (ax1, "Time series comparison"),
        (ax2, f"Difference (compare − {ref_col[:20]})"),
    ]:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(direction="in", length=3, width=0.6, labelsize=6.5)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.35,
                      color="#CCCCCC", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(ttl, fontsize=8, pad=3, loc="left", fontweight="bold")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax1.legend(fontsize=6, frameon=True, framealpha=0.85,
               edgecolor="#CCCCCC", loc="best", labelspacing=0.3)
    ax1.set_ylabel("Value", fontsize=7)
    ax2.set_ylabel("Difference", fontsize=7)
    ax2.set_xlabel("Year", fontsize=7)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


# ======================================================================
# main() — edit parameters here
# ======================================================================

def main() -> None:
    # Paths
    CSV = ("/g12338011ghq/project/hbb/OxygenDiffusion/"
           "Oxy_regression_Surface/results/Results_CSV/"
           "WOD_1900_2014_Ship.csv")
    DIR_OUT = os.path.dirname(CSV)

    df = load_csv(CSV)

    # ------------------------------------------------------------------
    # Use case 1: single-pair comparison (WOD reconstruction vs CMIP6 truth)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Use case 1: single-pair comparison")
    print("=" * 60)

    metrics = compute_metrics(
        y_true=to_annual(df)["anom_recon_median"],
        y_pred=to_annual(df)["anom_recon_model_src"],
        label_true="WOD reconstruction (median)",
        label_pred="Recon from model source",
        year_start=1950, year_end=2014,
    )
    if metrics:
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

    # ------------------------------------------------------------------
    # Use case 2: one vs many (WOD reconstruction vs reference products)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Use case 2: batch comparison")
    print("=" * 60)

    REF_COL = "anom_recon_median"
    COMPARE_COLS = [c for c in df.columns if c.startswith("anom_ref_")]

    if COMPARE_COLS:
        df_metrics = compute_metrics_batch(
            df=df, ref_col=REF_COL, compare_cols=COMPARE_COLS,
            year_start=1950, year_end=2014, resample="annual",
        )
        print_metrics_table(df_metrics)
        df_metrics.to_csv(
            os.path.join(DIR_OUT, "metrics_wod_vs_refs.csv"), index=False
        )

        plot_scatter_comparison(
            df=df, ref_col=REF_COL, compare_cols=COMPARE_COLS,
            year_start=1950, year_end=2014, resample="annual", dpi=300,
        )
        plot_timeseries_comparison(
            df=df, ref_col=REF_COL, compare_cols=COMPARE_COLS,
            year_start=1950, year_end=2014, resample="annual", dpi=300,
        )

    # ------------------------------------------------------------------
    # Use case 3: cross-CSV comparison (WOD vs IAP)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Use case 3: cross-CSV comparison (WOD vs IAP)")
    print("=" * 60)

    IAP_CSV = ("/g12338011ghq/project/hbb/OxygenDiffusion/"
               "Oxy_regression_Surface/results/Results_CSV/"
               "IAP_1940_2014_Ship.csv")

    if os.path.exists(IAP_CSV):
        df_iap = load_csv(IAP_CSV)
        iap_col = next(
            (c for c in df_iap.columns if "median" in c.lower()),
            df_iap.columns[0],
        )
        joint = pd.DataFrame({
            "WOD_recon": to_annual(df)["anom_recon_median"],
            "IAP_recon": to_annual(df_iap)[iap_col],
        })
        m = compute_metrics(
            joint["WOD_recon"], joint["IAP_recon"],
            label_true="WOD", label_pred="IAP",
            year_start=1950, year_end=2014,
        )
        if m:
            print("\n  WOD vs IAP (1950–2014, annual):")
            for k, v in m.items():
                print(f"    {k:<12}: {v}")


if __name__ == "__main__":
    main()
