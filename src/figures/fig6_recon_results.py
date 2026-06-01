"""
fig6_recon_results.py
=====================
Main reconstruction time-series with reference products (Fig 6).

Provides two plotting functions:
  - plot(): single dataset with CI and reference curves
  - plot_compare(): overlay multiple datasets with CI

All parameters are set in main() at the bottom.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional, List, Dict, Tuple

# Colour palette (Paul Tol, colourblind-friendly)
_TOLS_BRIGHT = [
    "#EE6677", "#4477AA", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB",
    "#000000", "#EE7733", "#0077BB", "#33BBEE", "#009988",
]

DEFAULT_STYLE = {
    "recon_color":   _TOLS_BRIGHT[0],
    "recon_lw":      2,
    "recon_label":   "This reconstruction (median)",
    "shade_alpha":   0.15,
    "truth_color":   "#222222",
    "truth_lw":      1.2,
    "truth_ls":      "--",
    "truth_label":   "CMIP6 truth",
    "model_src_color": "#CCBB44",
    "model_src_lw":  1.0,
    "model_src_ls":  "-.",
    "model_src_label": "Recon (model source)",
    "ref_colors":    _TOLS_BRIGHT[2:],
    "ref_lw":        1.0,
    "fs_title":      13,
    "fs_label":      12,
    "fs_tick":       11,
    "fs_legend":     11,
}

Y_SCALE = 1e6


# ======================================================================
# Data loading
# ======================================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV with DatetimeIndex."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    print(f"[load] {os.path.basename(csv_path)} -> "
          f"{df.shape[0]} rows x {df.shape[1]} cols")
    return df


# ======================================================================
# Style
# ======================================================================

def _apply_journal_style(ax, annual: bool, ylabel: str,
                         panel_title: str) -> None:
    st = DEFAULT_STYLE
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(0.6)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(axis="both", direction="in", length=3, width=0.6,
                   labelsize=st["fs_tick"])
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4,
                  color="#DDDDDD", alpha=0.8, zorder=0)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.4,
                  color="#DDDDDD", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=st["fs_label"], labelpad=4)
    ax.set_title(panel_title, fontsize=st["fs_title"], pad=4, loc="left")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    if annual:
        ax.set_xlabel("Year", fontsize=st["fs_label"], labelpad=4)
        ax.xaxis.set_major_locator(
            mticker.MaxNLocator(integer=True, nbins=8)
        )
    else:
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    leg = ax.legend(fontsize=st["fs_legend"], frameon=True,
                    framealpha=0.85, edgecolor="#CCCCCC",
                    borderpad=0.5, labelspacing=0.3,
                    handlelength=1.8, loc="upper left",
                    bbox_to_anchor=(1.01, 1), borderaxespad=0)
    leg.get_frame().set_linewidth(0.4)


# ======================================================================
# Column helpers
# ======================================================================

def _col(mode: str, name: str) -> str:
    return f"{mode}_{name}"


def resolve_ci_cols(mode: str, smooth: bool) -> Tuple[str, str, str]:
    tag = "smooth" if smooth else "raw"
    return (
        _col(mode, "recon_median"),
        _col(mode, f"recon_lo_{tag}"),
        _col(mode, f"recon_hi_{tag}"),
    )


def resolve_ref_cols(df: pd.DataFrame, mode: str,
                     ref_products: Optional[List[str]]) -> List[str]:
    """Resolve reference product column names."""
    prefix = f"{mode}_ref_"
    all_ref = [c for c in df.columns if c.startswith(prefix)]
    if not ref_products:
        return []
    if ref_products == ["all"]:
        return all_ref
    result = []
    for token in ref_products:
        matched = [c for c in all_ref if token.lower() in c.lower()]
        for c in matched:
            if c not in result:
                result.append(c)
    return result


def col_label(col: str) -> str:
    for prefix in ("anom_ref_", "abs_ref_", "anom_", "abs_"):
        col = col.replace(prefix, "")
    return col.replace("_", " ").strip()


# ======================================================================
# Drawing helpers
# ======================================================================

def _draw_panel(ax, df, x, median_col, lo_col, hi_col,
                show_shade, extra_cols, extra_colors, extra_styles,
                extra_labels, ylabel, panel_title, style, annual=False):
    """Draw a set of curves on a single Axes."""
    if show_shade and lo_col in df.columns and hi_col in df.columns:
        ax.fill_between(x, df[lo_col].values * Y_SCALE,
                        df[hi_col].values * Y_SCALE,
                        color=style["recon_color"],
                        alpha=style["shade_alpha"],
                        label="95% CI", zorder=1)

    if median_col in df.columns:
        mask = df[median_col].notna().values
        ax.plot(x[mask], df[median_col].values[mask] * Y_SCALE,
                color=style["recon_color"], lw=style["recon_lw"],
                label=style["recon_label"], zorder=3)

    for col in extra_cols:
        if col not in df.columns:
            continue
        mask = df[col].notna().values
        if not mask.any():
            continue
        lw, ls = extra_styles.get(col, (style["ref_lw"], "-"))
        ax.plot(x[mask], df[col].values[mask] * Y_SCALE,
                color=extra_colors[col], lw=lw, ls=ls,
                label=extra_labels.get(col, col_label(col)),
                alpha=0.88, zorder=2)

    if annual and median_col in df.columns:
        mask = df[median_col].notna().values
        ax.scatter(x[mask], df[median_col].values[mask],
                   color=style["recon_color"], s=22, zorder=4)

    _apply_journal_style(ax, annual=annual, ylabel=ylabel,
                         panel_title=panel_title)


def _draw_compare_panel(ax, datasets, mode, show_shade, annual,
                        ylabel, panel_title, trange,
                        ref_df=None, ref_cols=None,
                        ref_colors=None, ref_labels=None):
    """Overlay multiple datasets + optional reference products."""
    if ref_df is not None and ref_cols:
        df_ref = ref_df.copy()
        if trange:
            df_ref = df_ref.loc[f"{trange[0]}-01-01":f"{trange[1]}-12-31"]
        if annual:
            df_ref = df_ref.resample("YE").mean()
            x_ref = np.array(df_ref.index.year)
        else:
            x_ref = np.array(df_ref.index, dtype="datetime64[ms]")

        for idx, col in enumerate(ref_cols):
            if col not in df_ref.columns:
                continue
            mask = df_ref[col].notna().values
            if not mask.any():
                continue
            color = ref_colors[idx] if ref_colors else "gray"
            label_str = (ref_labels.get(col, col_label(col))
                         if ref_labels else col_label(col))
            ax.plot(x_ref[mask], df_ref[col].values[mask] * Y_SCALE,
                    color=color, lw=1.2, ls="--",
                    label=label_str, alpha=0.75, zorder=1)

    for ds in datasets:
        df = ds["df"].copy()
        if trange:
            df = df.loc[f"{trange[0]}-01-01":f"{trange[1]}-12-31"]
        if annual:
            df = df.resample("YE").mean()
            x = np.array(df.index.year)
        else:
            x = np.array(df.index, dtype="datetime64[ms]")

        smooth = ds.get("smooth_ci", False)
        color = ds["color"]
        label = ds["label"]
        lw = ds.get("lw", 2.0)
        ls = ds.get("ls", "-")
        do_shade = ds.get("shade", show_shade)
        shade_alpha = ds.get("shade_alpha", 0.15)

        median_col, lo_col, hi_col = resolve_ci_cols(mode, smooth)

        if do_shade and lo_col in df.columns and hi_col in df.columns:
            lo = df[lo_col].values
            hi = df[hi_col].values
            valid = ~(np.isnan(lo) | np.isnan(hi))
            if valid.any():
                ax.fill_between(x[valid], lo[valid] * Y_SCALE,
                                hi[valid] * Y_SCALE,
                                color=color, alpha=shade_alpha,
                                label=f"{label} 95% CI", zorder=2)

        if median_col in df.columns:
            mask = df[median_col].notna().values
            if mask.any():
                ax.plot(x[mask], df[median_col].values[mask] * Y_SCALE,
                        color=color, lw=lw, ls=ls,
                        label=label, zorder=3)

    _apply_journal_style(ax, annual=annual, ylabel=ylabel,
                         panel_title=panel_title)


# ======================================================================
# Main plotting API — single dataset
# ======================================================================

def plot(df: pd.DataFrame, mode: str = "anom",
         show_monthly: bool = True, show_annual: bool = True,
         show_shade: bool = False, smooth_ci: bool = False,
         show_truth: bool = True, show_model_src: bool = False,
         ref_products: Optional[List[str]] = None,
         trange: Optional[Tuple[int, int]] = None,
         title: Optional[str] = None, ylabel: Optional[str] = None,
         style: Optional[Dict] = None, figsize: Optional[Tuple] = None,
         out: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    """Plot reconstruction time-series with optional reference curves."""
    st = {**DEFAULT_STYLE, **(style or {})}

    df = df.copy()
    if trange:
        df = df.loc[f"{trange[0]}-01-01":f"{trange[1]}-12-31"]

    median_col, lo_col, hi_col = resolve_ci_cols(mode, smooth_ci)
    ref_cols = resolve_ref_cols(df, mode, ref_products)

    extra_cols, extra_colors, extra_styles, extra_labels = [], {}, {}, {}

    truth_col = _col(mode, "cmip6_truth")
    if show_truth and truth_col in df.columns:
        extra_cols.append(truth_col)
        extra_colors[truth_col] = st["truth_color"]
        extra_styles[truth_col] = (st["truth_lw"], st["truth_ls"])
        extra_labels[truth_col] = st["truth_label"]

    src_col = _col(mode, "recon_model_src")
    if show_model_src and src_col in df.columns:
        extra_cols.append(src_col)
        extra_colors[src_col] = st["model_src_color"]
        extra_styles[src_col] = (st["model_src_lw"], st["model_src_ls"])
        extra_labels[src_col] = st["model_src_label"]

    for idx, col in enumerate(ref_cols):
        extra_cols.append(col)
        extra_colors[col] = st["ref_colors"][idx % len(st["ref_colors"])]
        extra_styles[col] = (st["ref_lw"], "-")

    mode_str = "Anomaly" if mode == "anom" else "Absolute"
    ylabel_str = ylabel or (
        "Mean DO Anomaly (mol/m3)" if mode == "anom" else "Mean DO (mol/m3)"
    )
    title_str = title or f"Surface DO {mode_str} — Reconstruction"

    n_panels = int(show_monthly) + int(show_annual)
    if n_panels == 0:
        raise ValueError("At least one panel required.")

    fig, axes = plt.subplots(n_panels, 1,
                              figsize=figsize or (7.0, 3.2 * n_panels),
                              squeeze=False)
    fig.suptitle(title_str, fontsize=DEFAULT_STYLE["fs_title"],
                 y=1.02, fontweight="bold")

    panel = 0
    if show_monthly:
        x_m = np.array(df.index, dtype="datetime64[ms]")
        _draw_panel(axes[panel][0], df, x_m,
                    median_col, lo_col, hi_col, show_shade,
                    extra_cols, extra_colors, extra_styles,
                    extra_labels, ylabel_str, "Monthly mean", st,
                    annual=False)
        panel += 1

    if show_annual:
        df_yr = df.resample("YE").mean()
        x_yr = np.array(df_yr.index.year)
        _draw_panel(axes[panel][0], df_yr, x_yr,
                    median_col, lo_col, hi_col, show_shade,
                    extra_cols, extra_colors, extra_styles,
                    extra_labels, ylabel_str, "Annual mean", st,
                    annual=True)

    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


# ======================================================================
# Main plotting API — multi-dataset comparison
# ======================================================================

def plot_compare(datasets: List[Dict], mode: str = "anom",
                 show_monthly: bool = True, show_annual: bool = True,
                 show_shade: bool = True,
                 ref_source: Optional[pd.DataFrame] = None,
                 ref_products: Optional[List[str]] = None,
                 ref_colors: Optional[List[str]] = None,
                 ref_labels: Optional[Dict[str, str]] = None,
                 trange: Optional[Tuple[int, int]] = None,
                 title: Optional[str] = None, ylabel: Optional[str] = None,
                 figsize: Optional[Tuple] = None,
                 out: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    """Overlay multiple reconstruction datasets on the same figure."""
    if not datasets:
        raise ValueError("datasets list cannot be empty.")

    n_panels = int(show_monthly) + int(show_annual)
    if n_panels == 0:
        raise ValueError("At least one panel required.")

    resolved_ref_cols, resolved_ref_colors = [], []
    if ref_source is not None and ref_products:
        resolved_ref_cols = resolve_ref_cols(ref_source, mode, ref_products)
        pool = ref_colors or DEFAULT_STYLE["ref_colors"]
        resolved_ref_colors = [pool[i % len(pool)]
                               for i in range(len(resolved_ref_cols))]

    mode_str = "Anomaly" if mode == "anom" else "Absolute"
    ylabel_str = ylabel or (
        "Mean DO Anomaly (mol/m3)" if mode == "anom" else "Mean DO (mol/m3)"
    )
    labels = " vs. ".join(ds["label"] for ds in datasets)
    title_str = title or f"Surface DO {mode_str} — {labels}"

    fig, axes = plt.subplots(n_panels, 1,
                              figsize=figsize or (18, 6 * n_panels),
                              squeeze=False)
    fig.suptitle(title_str, fontsize=DEFAULT_STYLE["fs_title"],
                 y=1.02, fontweight="bold")

    panel = 0
    if show_monthly:
        _draw_compare_panel(axes[panel][0], datasets, mode, show_shade,
                            False, ylabel_str, "Monthly mean", trange,
                            ref_source, resolved_ref_cols,
                            resolved_ref_colors, ref_labels)
        panel += 1
    if show_annual:
        _draw_compare_panel(axes[panel][0], datasets, mode, show_shade,
                            True, ylabel_str, "Annual mean", trange,
                            ref_source, resolved_ref_cols,
                            resolved_ref_colors, ref_labels)

    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


# ======================================================================
# main()
# ======================================================================

def main():
    DIR = ("/g12338011ghq/project/hbb/OxygenDiffusion/"
           "Oxy_regression_Surface/results/Results_CSV")
    DIR1 = ("/g12338011ghq/project/hbb/OxygenDiffusion/"
            "Oxy_regression_Surface/results/5deg_experiment")

    df_ref = load_csv(os.path.join(DIR, "WOD_1900_2014_Ship.csv"))
    df_wod = load_csv(os.path.join(DIR, "Ship_bias_LR_2000_2014_monthly.csv"))
    df_iap = load_csv(os.path.join(DIR, "IAP_1940_2014_Ship.csv"))

    plot_compare(
        datasets=[
            {
                "df": df_iap,
                "label": "IAP (Ship)",
                "color": "#4477AA",
                "lw": 2.5, "ls": "-",
                "shade": False,
                "smooth_ci": True,
            },
            {
                "df": df_wod,
                "label": "WOD (Ship)",
                "color": "#EE6677",
                "lw": 2.5, "ls": "-",
                "shade": True, "shade_alpha": 0.15,
                "smooth_ci": True,
            },
        ],
        mode="anom",
        show_monthly=False,
        show_annual=True,
        show_shade=False,
        ref_source=df_ref,
        trange=(1950, 2014),
        ref_products=["GT-OI", "GT-ML", "SJTU-JW", "SJTU-HZ",
                      "UTAS", "UW", "anom_ref_MMM"],
        out=os.path.join(
            os.path.dirname(DIR),
            "FigureResults/fig6_recon_results.png"
        ),
        figsize=(12, 6.2),
        ylabel="Dissolved Oxygen (μmol m⁻³)",
        title="Surface DO Anomaly",
        dpi=300,
    )


if __name__ == "__main__":
    main()
