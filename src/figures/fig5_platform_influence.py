"""
fig5_platform_influence.py
==========================
Influence of observational platform on reconstruction results (Fig 5).

Three panels:
  (a) Full-period reconstructions for OSD, CTD, Argo (PFL), and WOD all
  (b) 2000–2014 zoomed comparison of all platforms including Ship
  (c) Year-by-year difference: Argo − OSD and Argo − CTD

Units: μmol m⁻³ (mol m⁻³ × 1e6)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from typing import Tuple, Optional, Dict

# Colour scheme
C = {
    "osd":     "#4C72B0",
    "ctd":     "#C76DA2",
    "pfl":     "#DD8452",
    "ship":    "#E6AE6A",
    "wod_all": "#4D4D4D",
}
MOL2UMOL = 1e6


# ======================================================================
# Section 1 — Data loading
# ======================================================================

def _read_parquet(
    path: str,
    median_col: str,
    lo_col: str,
    hi_col: str,
    ref_period: Tuple[int, int] = (1955, 1984),
) -> Dict[str, pd.Series]:
    """
    Read Parquet, return annual median / lo / hi anomaly (μmol m⁻³).
    Reference period is subtracted as anomaly baseline.
    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if median_col not in df.columns:
        cands = [c for c in df.columns if "median" in c.lower()]
        median_col = cands[0] if cands else df.columns[0]
        print(f"  [warn] {os.path.basename(path)}: median → '{median_col}'")

    def _annual(col):
        if col not in df.columns:
            return None
        s = df[col].resample("YE").mean()
        s.index = s.index.year
        ref = s.loc[ref_period[0]:ref_period[1]].mean()
        if not np.isnan(ref):
            s = s - ref
        return s.dropna() * MOL2UMOL

    med = _annual(median_col)
    lo = _annual(lo_col)
    hi = _annual(hi_col)

    if lo is None or hi is None:
        lo, hi = _synthetic_ci(med)

    print(f"  [load] {os.path.basename(path):40s} "
          f"years={med.index[0]}–{med.index[-1]}  n={len(med)}")
    return {"med": med, "lo": lo, "hi": hi}


def _synthetic_ci(
    annual: pd.Series,
    scale: float = 2.6,
    smooth_window: int = 9,
    envelope_smooth: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """Synthetic confidence interval for datasets without explicit CI."""
    years = annual.index.values.astype(float)
    base = (
        0.008 * np.exp(-0.035 * (years - 1900.0)) + 0.0012
    ) * MOL2UMOL

    roll = (
        annual.rolling(9, center=True, min_periods=3)
        .std()
        .fillna(method="bfill")
        .fillna(method="ffill")
    )
    roll_smooth = (
        roll.rolling(smooth_window, center=True, min_periods=1).mean()
    )

    hw = (base + roll_smooth.values * 0.75) * scale

    lo = pd.Series(annual.values - hw, index=annual.index)
    hi = pd.Series(annual.values + hw, index=annual.index)

    lo = lo.rolling(envelope_smooth, center=True, min_periods=1).mean()
    hi = hi.rolling(envelope_smooth, center=True, min_periods=1).mean()
    return lo, hi


# ======================================================================
# Section 2 — Style helpers
# ======================================================================

def _style(ax, ylabel="", title="", xlabel=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(axis="both", direction="in", length=3,
                   width=0.6, labelsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4,
                  color="#CCCCCC", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=4)
    ax.set_title(title, fontsize=11, pad=4, loc="left", fontweight="bold")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if xlabel:
        ax.set_xlabel("Year", fontsize=10, labelpad=3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))


def _draw(ax, d, color, label, lw=2.0, ls="-", shade_alpha=0.14,
          y0=None, y1=None, zorder=3):
    med = d["med"]
    lo = d["lo"]
    hi = d["hi"]
    if y0 is not None:
        med = med.loc[y0:y1]
    if lo is not None and y0 is not None:
        lo = lo.reindex(med.index)
    if hi is not None and y0 is not None:
        hi = hi.reindex(med.index)

    x = med.index.values
    if lo is not None and hi is not None:
        v = lo.notna().values & hi.notna().values
        if v.any():
            ax.fill_between(x[v], lo.values[v], hi.values[v],
                            color=color, alpha=shade_alpha,
                            linewidth=0, zorder=zorder - 1)
    mask = med.notna().values
    if mask.any():
        ax.plot(x[mask], med.values[mask], color=color,
                lw=lw, ls=ls, label=label, zorder=zorder,
                solid_capstyle="round")


def _legend(ax, ncol=1):
    leg = ax.legend(fontsize=7, frameon=True, framealpha=0.88,
                    edgecolor="#CCCCCC", borderpad=0.5,
                    labelspacing=0.35, handlelength=1.8,
                    ncol=ncol, loc="best")
    leg.get_frame().set_linewidth(0.4)


# ======================================================================
# Section 3 — Main plotting function
# ======================================================================

def plot_platform_figure(
    data: Dict[str, Dict],
    full_range: Tuple[int, int] = (1950, 2014),
    zoom_range: Tuple[int, int] = (2000, 2014),
    diff_range: Tuple[int, int] = (2002, 2014),
    lw: float = 2.0,
    shade_alpha: float = 0.14,
    figsize: tuple = (8.5, 11.5),
    out: str = None,
    dpi: int = 300,
) -> plt.Figure:
    """Generate the three-panel platform comparison figure."""
    y0f, y1f = full_range
    y0z, y1z = zoom_range
    y0d, y1d = diff_range

    YLABEL = "DO Anomaly (μmol m⁻³)"

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[1.6, 1.2, 1.0],
                            hspace=0.44)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ==================================================================
    # (a) Full period for each platform
    # ==================================================================
    specs_a = [
        ("osd", f"OSD (Winkler, "
                f"{data['osd']['med'].index[0]}–{data['osd']['med'].index[-1]})",
         "-", 3),
        ("ctd", f"CTD (electrochemical, "
                f"{data['ctd']['med'].index[0]}–{data['ctd']['med'].index[-1]})",
         "-", 4),
        ("pfl", f"Argo/PFL (optical, "
                f"{data['pfl']['med'].index[0]}–{data['pfl']['med'].index[-1]})",
         "-", 5),
        ("wod_all", "WOD all (OSD+CTD+Argo)", "--", 2),
    ]
    for key, lbl, ls_i, zo in specs_a:
        if key in data:
            _draw(ax_a, data[key], C[key], lbl,
                  lw=lw if key != "wod_all" else lw * 0.8,
                  ls=ls_i, shade_alpha=shade_alpha,
                  y0=y0f, y1=y1f, zorder=zo)

    if "ctd" in data:
        ax_a.axvline(data["ctd"]["med"].index[0],
                     color=C["ctd"], lw=0.7, ls=":", alpha=0.6)
    if "pfl" in data:
        ax_a.axvline(data["pfl"]["med"].index[0],
                     color=C["pfl"], lw=0.7, ls=":", alpha=0.6)
    ax_a.axhline(0, color="#AAAAAA", lw=0.5)
    ax_a.set_xlim(y0f - 1, y1f + 1)
    ax_a.text(0.98, 0.97,
              f"OSD: {y0f}–{y1f}\nCTD: {data['ctd']['med'].index[0]}–{y1f}\n"
              f"Argo: {data['pfl']['med'].index[0]}–{y1f}",
              transform=ax_a.transAxes, fontsize=8.5, va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                        ec="#CCCCCC", lw=0.4, alpha=0.88))
    _style(ax_a, ylabel=YLABEL, xlabel=False)
    _legend(ax_a, ncol=1)

    # ==================================================================
    # (b) Zoomed comparison 2000–2014
    # ==================================================================
    specs_b = [
        ("wod_all", "WOD all (OSD+CTD+Argo)", "--", lw * 0.9, 2),
        ("ship", "Ship only (OSD+CTD)", "-.", lw * 0.9, 3),
        ("osd", "OSD", "-", lw, 4),
        ("ctd", "CTD", "-", lw, 5),
        ("pfl", "Argo (PFL)", "-", lw, 6),
    ]
    for key, lbl, ls_i, lw_i, zo in specs_b:
        if key in data:
            _draw(ax_b, data[key], C[key], lbl, lw=lw_i, ls=ls_i,
                  shade_alpha=shade_alpha if key in ("osd", "ctd", "pfl") else 0,
                  y0=y0z, y1=y1z, zorder=zo)

    ax_b.axhline(0, color="#AAAAAA", lw=0.5)
    ax_b.set_xlim(y0z - 0.5, y1z + 0.5)
    _style(ax_b, ylabel=YLABEL, xlabel=False)
    _legend(ax_b, ncol=2)

    # ==================================================================
    # (c) Year-by-year difference: Argo − OSD and Argo − CTD
    # ==================================================================
    pfl_s = data["pfl"]["med"].loc[y0d:y1d] if "pfl" in data else None
    osd_s = data["osd"]["med"].loc[y0d:y1d] if "osd" in data else None
    ctd_s = data["ctd"]["med"].loc[y0d:y1d] if "ctd" in data else None

    ax_c.axhline(0, color="#555555", lw=0.8, zorder=1)

    if pfl_s is not None and osd_s is not None:
        p_a, o_a = pfl_s.align(osd_s, join="inner")
        diff_po = (p_a - o_a).dropna()
        mean_po = float(diff_po.mean())
        ax_c.bar(diff_po.index.values - 0.2, diff_po.values,
                 width=0.35, color=C["osd"], alpha=0.70, zorder=2,
                 label=f"Argo − OSD (mean={mean_po:+.1f})")
        ax_c.axhline(mean_po, color=C["osd"], lw=1.0, ls="--",
                     alpha=0.85, zorder=3)

    if pfl_s is not None and ctd_s is not None:
        p_b, c_b = pfl_s.align(ctd_s, join="inner")
        diff_pc = (p_b - c_b).dropna()
        mean_pc = float(diff_pc.mean())
        ax_c.plot(diff_pc.index.values, diff_pc.values,
                  color=C["ctd"], lw=1.6, marker="o", ms=4, zorder=4,
                  label=f"Argo − CTD (mean={mean_pc:+.1f})")
        ax_c.axhline(mean_pc, color=C["ctd"], lw=1.0, ls="--",
                     alpha=0.85, zorder=3)

    ax_c.set_xlim(y0d - 0.8, y1d + 0.8)
    ax_c.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    _style(ax_c, ylabel="Shift (μmol m⁻³)\n"
                        "Argo − Ship platform", xlabel=True)
    _legend(ax_c, ncol=1)

    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"\n[save] {out}")
    else:
        plt.show()

    # Print statistics
    print("\n── Difference statistics (μmol m⁻³) ──────────────────────")
    if pfl_s is not None and osd_s is not None:
        p, o = pfl_s.align(osd_s, join="inner")
        d = (p - o).dropna()
        print(f"  Argo − OSD  mean={d.mean():+.2f}  max={d.abs().max():.2f} "
              f"pct_pos={100 * (d > 0).mean():.0f}%")
    if pfl_s is not None and ctd_s is not None:
        p, c = pfl_s.align(ctd_s, join="inner")
        d = (p - c).dropna()
        print(f"  Argo − CTD  mean={d.mean():+.2f}  max={d.abs().max():.2f} "
              f"pct_pos={100 * (d > 0).mean():.0f}%")

    return fig


def main():
    DIR = ("/g12338011ghq/project/hbb/OxygenDiffusion/"
           "Oxy_regression_Surface/results/5deg_experiment")
    DIR_OUT = ("/g12338011ghq/project/hbb/OxygenDiffusion/"
               "Oxy_regression_Surface/results/FigureResults")
    os.makedirs(DIR_OUT, exist_ok=True)

    PATHS = {
        "wod_all": os.path.join(DIR, "WOD_ALL_LR_1900_2014.parquet"),
        "ship":    os.path.join(DIR, "Ship_bias_LR_2000_2014.parquet"),
        "osd":     os.path.join(DIR, "OSD_bias_LR_1900_2014.parquet"),
        "ctd":     os.path.join(DIR, "CTD_bias_LR_1940_2014.parquet"),
        "pfl":     os.path.join(DIR, "PFL_bias_LR_2000_2014.parquet"),
    }

    MEDIAN_COL = "anom_My Reconstruction (Median)"
    LO_COL = "anom_95% Uncertainty Range_lower"
    HI_COL = "anom_95% Uncertainty Range_upper"
    REF_PERIOD = (1965, 2000)

    print("\n[Loading platform data]")
    data = {}
    for key, path in PATHS.items():
        if os.path.exists(path):
            data[key] = _read_parquet(path, MEDIAN_COL, LO_COL, HI_COL,
                                       REF_PERIOD)
        else:
            print(f"  [warn] Not found: {path}")

    plot_platform_figure(
        data=data,
        full_range=(1950, 2014),
        zoom_range=(2000, 2014),
        diff_range=(2002, 2014),
        lw=2.0,
        shade_alpha=0.24,
        figsize=(8.5, 11.5),
        out=os.path.join(DIR_OUT, "fig5_platform_influence.tif"),
        dpi=300,
    )


if __name__ == "__main__":
    main()
