"""
fig8_model_families.py
======================
WOD reconstruction vs. CMIP6 model families (Fig 8).

Shows the reconstruction (median + 95% CI) overlaid on the spread of
CMIP6 model families (shaded percentile ranges), with all-model MMM.

Also produces a supplementary panel showing deoxygenation rate
distribution by model family.
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from typing import Dict, List, Tuple, Optional

# Colour per model family
FAMILY_COLORS = {
    "MPI-ESM1-2-LR":   "#4477AA",
    "MPI-ESM1-2-HR":   "#66CCEE",
    "MPI-ESM-1-2-HAM": "#228833",
    "ACCESS-ESM1-5":   "#CCBB44",
    "CNRM-ESM2-1":     "#AA3377",
    "CanESM5":         "#EE6677",
    "MMM":             "#222222",
    "WOD":             "#BB0000",
}

FAMILY_PREFIXES = {
    "MPI-ESM1-2-LR":   "MPI-ESM1-2-LR",
    "MPI-ESM1-2-HR":   "MPI-ESM1-2-HR",
    "MPI-ESM-1-2-HAM": "MPI-ESM-1-2-HAM",
    "ACCESS-ESM1-5":   "ACCESS-ESM1-5",
    "CNRM-ESM2-1":     "CNRM-ESM2-1",
    "CanESM5":         "CanESM5",
}


# ======================================================================
# Data loading
# ======================================================================

def load_recon(
    csv_path: str,
    median_col: str = "anom_recon_median",
    lo_col: str = "anom_recon_lo_smooth",
    hi_col: str = "anom_recon_hi_smooth",
    year_start: int = 1950,
    year_end: int = 2014,
    ref_period: Tuple = (1955, 1984),
) -> Dict[str, pd.Series]:
    """Load reconstruction CSV, return median / lo / hi annual series."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if median_col not in df.columns:
        cands = [c for c in df.columns if "median" in c.lower()]
        median_col = cands[0] if cands else df.columns[0]
        print(f"[warn] recon median -> using '{median_col}'")

    def _annual(col):
        if col not in df.columns:
            return None
        s = df[col].resample("YE").mean()
        s.index = s.index.year
        s = s.loc[year_start:year_end].dropna()
        ref = s.loc[ref_period[0]:ref_period[1]].mean()
        return s - ref

    res = {"median": _annual(median_col),
           "lo":     _annual(lo_col),
           "hi":     _annual(hi_col)}
    print(f"[recon] years={res['median'].index[0]}–{res['median'].index[-1]}")
    return res


def load_single_model(fp: str, var_name: str, time_range: Tuple,
                      ref_period: Tuple) -> Optional[pd.Series]:
    """Load a single model's annual global-mean series."""
    ts = slice(f"{time_range[0]}-01-01", f"{time_range[1]}-12-31")
    try:
        with xr.open_dataset(fp, chunks=None, decode_times=False) as ds:
            ds = xr.decode_cf(ds, use_cftime=True)
            if isinstance(ds.time.to_index(),
                          xr.coding.cftimeindex.CFTimeIndex):
                ds["time"] = ds.indexes["time"].to_datetimeindex(unsafe=True)
            ds_s = ds.sel(time=ts)
            if ds_s.time.size == 0:
                return None
            da = ds_s[var_name]
            depth_dim = next((d for d in ("lev", "olevel", "depth")
                              if d in da.dims), None)
            if depth_dim:
                da = da.isel({depth_dim: 0})
            lat = next((l for l in ("lat", "latitude") if l in da.dims),
                       None)
            lon = next((l for l in ("lon", "longitude") if l in da.dims),
                       None)
            if lat is None:
                return None
            w = np.cos(np.deg2rad(da[lat]))
            w.name = "weights"
            monthly = da.weighted(w).mean(
                [d for d in (lat, lon) if d]
            ).load()
            annual = monthly.to_series()
            annual.index = pd.to_datetime(annual.index)
            annual = annual.resample("YE").mean()
            annual.index = annual.index.year
            ref = annual.loc[ref_period[0]:ref_period[1]].mean()
            if np.isnan(ref):
                return None
            return annual - ref
    except Exception as e:
        print(f"  [err] {os.path.basename(fp)}: {e}")
        return None


def load_families(
    model_dirs: List[str], var_name: str,
    time_range: Tuple, ref_period: Tuple,
    year_start: int, year_end: int,
) -> Dict[str, List[pd.Series]]:
    """Load and group model members by family."""
    files = sorted(set(
        f for d in model_dirs
        for f in glob.glob(os.path.join(d, "**", "*.nc"), recursive=True)
    ))
    print(f"[scan] {len(files)} NetCDF files")

    family_data = {k: [] for k in FAMILY_PREFIXES}

    for fp in files:
        name = os.path.basename(fp).replace(".nc", "")
        fam = next((k for k, p in FAMILY_PREFIXES.items()
                    if name.startswith(p)), None)
        if fam is None:
            continue
        s = load_single_model(fp, var_name, time_range, ref_period)
        if s is not None:
            s = s.loc[year_start:year_end].dropna()
            if len(s) >= 10:
                family_data[fam].append(s)

    for fam, m in family_data.items():
        print(f"  {fam:<20}: {len(m)} members")
    return family_data


# ======================================================================
# Family statistics
# ======================================================================

def family_stats(members: List[pd.Series], lo_pct: float = 10.0,
                 hi_pct: float = 90.0) -> Dict:
    """Compute family mean / median / percentile range."""
    if not members:
        return {}
    df = pd.DataFrame({i: s for i, s in enumerate(members)}).dropna(how="all")
    return {
        "mean":   df.mean(axis=1),
        "median": df.median(axis=1),
        "lo":     df.quantile(lo_pct / 100, axis=1),
        "hi":     df.quantile(hi_pct / 100, axis=1),
        "n":      df.notna().sum(axis=1),
    }


# ======================================================================
# Style
# ======================================================================

def _style(ax, ylabel="", title=""):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(axis="both", direction="in", length=3, width=0.6,
                   labelsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4,
                  color="#CCCCCC", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Year", fontsize=10, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=4)
    ax.set_title(title, fontsize=11, pad=5, loc="left", fontweight="bold")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))


# ======================================================================
# Main plotting function
# ======================================================================

def plot_recon_vs_families(
    recon: Dict, family_data: Dict,
    shade_lo: float = 10.0, shade_hi: float = 90.0,
    use_mean: bool = True, show_mmm: bool = True,
    show_annual: bool = True, show_monthly: bool = False,
    recon_lw: float = 2.0, recon_shade_alpha: float = 0.20,
    family_lw: float = 1.4, family_shade_alpha: float = 0.12,
    mmm_lw: float = 1.6,
    ylabel: str = "DO Anomaly (mol m⁻³) relative to 1955–1984",
    title: str = "WOD reconstruction vs. CMIP6 model families",
    figsize: tuple = (9.0, 6.0),
    out: str = None, dpi: int = 300,
) -> plt.Figure:
    """
    Main figure: reconstruction (red) overlaid on CMIP6 family ranges.

    Layer order (bottom to top):
      1. Family percentile shading
      2. All-model MMM (black dashed)
      3. Family mean lines
      4. Reconstruction CI (red shading)
      5. Reconstruction median (red line, top)
    """
    n_panels = int(show_monthly) + int(show_annual)
    if n_panels == 0:
        raise ValueError("At least one panel required.")

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=12, y=1.01, fontweight="bold")

    fam_stats = {
        fam: family_stats(members, shade_lo, shade_hi)
        for fam, members in family_data.items() if members
    }

    all_members = [m for members in family_data.values() for m in members]
    if all_members and show_mmm:
        mmm = pd.DataFrame({i: s for i, s in enumerate(all_members)}).mean(axis=1)
    else:
        mmm = None

    panel = 0
    for annual in [False, True]:
        if annual and not show_annual:
            continue
        if not annual and not show_monthly:
            continue

        ax = axes[panel][0]

        # Layer 1: family shading
        for fam, st in fam_stats.items():
            if not st:
                continue
            color = FAMILY_COLORS.get(fam, "#AAAAAA")
            if annual:
                valid = ~(np.isnan(st["lo"].values) |
                          np.isnan(st["hi"].values))
                if valid.any():
                    ax.fill_between(st["lo"].index.values[valid],
                                    st["lo"].values[valid],
                                    st["hi"].values[valid],
                                    color=color, alpha=family_shade_alpha,
                                    zorder=1)

        # Layer 2: MMM
        if mmm is not None:
            ax.plot(mmm.index.values, mmm.values,
                    color=FAMILY_COLORS["MMM"], lw=mmm_lw, ls="--",
                    zorder=4,
                    label=f"All-model MMM (n={len(all_members)})")

        # Layer 3: family mean lines
        for fam, st in fam_stats.items():
            if not st:
                continue
            color = FAMILY_COLORS.get(fam, "#AAAAAA")
            ctr = st["mean"] if use_mean else st["median"]
            n_mem = int(st["n"].iloc[0]) if len(st["n"]) > 0 else 0
            mask = ctr.notna().values
            ax.plot(ctr.index.values[mask], ctr.values[mask],
                    color=color, lw=family_lw, zorder=5,
                    label=f"{fam} (n={n_mem})")

        # Layer 4: reconstruction CI
        med, lo, hi = recon.get("median"), recon.get("lo"), recon.get("hi")
        if lo is not None and hi is not None:
            r_lo, r_hi = lo.align(hi, join="inner")
            r_med, _ = med.align(r_lo, join="inner")
            valid = r_lo.notna().values & r_hi.notna().values
            ax.fill_between(r_lo.index.values[valid],
                            r_lo.values[valid], r_hi.values[valid],
                            color=FAMILY_COLORS["WOD"],
                            alpha=recon_shade_alpha,
                            zorder=6, label="WOD 95% CI")

        # Layer 5: reconstruction median
        mask = med.notna().values
        ax.plot(med.index.values[mask], med.values[mask],
                color=FAMILY_COLORS["WOD"], lw=recon_lw,
                zorder=7, label="WOD reconstruction")
        ax.axhline(0, color="#888888", lw=0.5, zorder=0)

        _style(ax, ylabel=ylabel, title="Annual mean" if annual else "Monthly mean")
        leg = ax.legend(fontsize=9.5, frameon=True, framealpha=0.88,
                        edgecolor="#CCCCCC", borderpad=0.5,
                        labelspacing=0.35, handlelength=1.6,
                        ncol=2, loc="lower left")
        leg.get_frame().set_linewidth(0.4)
        panel += 1

    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


def plot_rate_bias_summary(
    family_data: Dict[str, List[pd.Series]],
    recon_rate: float,
    out: str = None, dpi: int = 300,
) -> Optional[plt.Figure]:
    """Box-plot of deoxygenation rates by CMIP6 family vs reconstruction."""
    MOL_TO_UMOL, YRS_PER_DEC = 1e6, 10.0

    def _rate(s):
        ok = s.dropna()
        if len(ok) < 10:
            return np.nan
        slope, *_ = stats.linregress(ok.index.astype(float), ok.values)
        return slope * MOL_TO_UMOL * YRS_PER_DEC

    fam_rates = {}
    for fam, members in family_data.items():
        rates = [r for r in [_rate(s) for s in members] if not np.isnan(r)]
        if rates:
            fam_rates[fam] = np.array(rates)

    if not fam_rates:
        return None

    families = sorted(fam_rates, key=lambda f: np.median(fam_rates[f]))
    fig, ax = plt.subplots(figsize=(max(5.5, len(families) * 1.1), 4.5))

    positions = list(range(len(families)))
    data = [fam_rates[f] for f in families]
    bp = ax.boxplot(data, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", ms=3, alpha=0.4,
                                   markerfacecolor="#AAAAAA",
                                   markeredgecolor="none"),
                    medianprops=dict(color="#222222", lw=1.8),
                    boxprops=dict(edgecolor="#AAAAAA", lw=0.8),
                    whiskerprops=dict(color="#AAAAAA", lw=0.8, ls="--"),
                    capprops=dict(color="#AAAAAA", lw=0.8))

    for patch, fam in zip(bp["boxes"], families):
        c = FAMILY_COLORS.get(fam, "#AAAAAA")
        patch.set_facecolor(c)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(42)
    for pi, (fam, vals) in enumerate(zip(families, data)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        color = FAMILY_COLORS.get(fam, "#AAAAAA")
        ax.scatter(np.full(len(vals), pi) + jitter, vals, s=18,
                   color=color, alpha=0.75, edgecolors="white",
                   linewidths=0.3, zorder=3)

    ax.axhline(recon_rate, color=FAMILY_COLORS["WOD"], lw=1.5, ls="--",
               zorder=4,
               label=f"WOD reconstruction ({recon_rate:+.0f} μmol m⁻³ dec⁻¹)")
    ax.axhline(0, color="#AAAAAA", lw=0.5, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(families, fontsize=7, rotation=15, ha="right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(axis="both", direction="in", length=3, width=0.6,
                   labelsize=7)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4,
                  color="#CCCCCC", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Deoxygenation rate (μmol m⁻³ dec⁻¹)", fontsize=8)
    ax.set_title("Deoxygenation rate by CMIP6 model family\n"
                 "(below WOD line = model overestimates deoxygenation)",
                 fontsize=8, loc="left", fontweight="bold")

    leg = ax.legend(fontsize=7, frameon=True, framealpha=0.88,
                    edgecolor="#CCCCCC", loc="upper right")
    leg.get_frame().set_linewidth(0.4)
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


def main():
    CSV_RECON = ("/g12338011ghq/project/hbb/OxygenDiffusion/"
                 "Oxy_regression_Surface/results/Results_CSV/"
                 "WOD_1900_2014_Ship.csv")

    MODEL_DIRS = [
        "/g12338011ghq/project/hbb/Dataset/CMIP6_DO_GlobalSurface_5deg/"
        + m for m in
        ["MPI-ESM1-2-LR", "CanESM5", "MPI-ESM1-2-HR",
         "CNRM-ESM2-1", "ACCESS-ESM1-5", "MPI-ESM-1-2-HAM"]
    ]

    VAR_NAME = "o2"
    TIME_RANGE = (1950, 2014)
    REF_PERIOD = (1950, 2014)

    recon = load_recon(CSV_RECON, year_start=1950, year_end=2014,
                       ref_period=REF_PERIOD)
    family_data = load_families(MODEL_DIRS, VAR_NAME, TIME_RANGE,
                                REF_PERIOD, 1950, 2014)

    # Main figure
    plot_recon_vs_families(
        recon=recon, family_data=family_data,
        shade_lo=10.0, shade_hi=90.0, use_mean=True, show_mmm=True,
        show_annual=True, show_monthly=False,
        recon_lw=2.2, recon_shade_alpha=0.22,
        family_lw=1.5, family_shade_alpha=0.12,
        ylabel="DO Anomaly (mol m⁻³) relative to 1955–1984",
        title=f"WOD reconstruction vs. CMIP6 model families (1950–2014)",
        figsize=(11.0, 5.5),
        out=("/g12338011ghq/project/hbb/OxygenDiffusion/"
             "Oxy_regression_Surface/results/FigureResults/"
             "fig8_recon_vs_families.png"),
        dpi=300,
    )

    # Rate summary panel
    med = recon["median"].dropna()
    slope_recon, *_ = stats.linregress(med.index.astype(float), med.values)
    recon_rate = slope_recon * 1e6 * 10
    print(f"\n[recon rate] {recon_rate:+.1f} μmol m⁻³ decade⁻¹")

    plot_rate_bias_summary(
        family_data=family_data, recon_rate=recon_rate,
        out=("/g12338011ghq/project/hbb/OxygenDiffusion/"
             "Oxy_regression_Surface/results/FigureResults/"
             "fig8_family_rate_distribution.png"),
        dpi=300,
    )


if __name__ == "__main__":
    main()
