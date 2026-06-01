"""
fig10_cmip6_comparison.py
=========================
Systematic comparison between CMIP6 multi-model ensemble (MMM) and
observational reconstructions (WOD, IAP) — Fig 10.

Panels:
  (A) CMIP6 inter-model spread (annual)
  (B) MMM ± 1σ vs WOD & IAP reconstructions (monthly + annual)
  (C) Annual bias time-series (recon − MMM)
  (D) CMIP6 spread vs reconstruction uncertainty width
  (E) Deoxygenation rate: CMIP6 ensemble vs reconstructions

All parameters in main().
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from scipy import stats
from typing import Dict, List, Optional, Tuple

# Colour palette
_C = {
    "wod":    "#EE6677",
    "iap":    "#4477AA",
    "mmm":    "#222222",
    "spread": "#AAAAAA",
    "models": "#CCCCCC",
    "diff":   "#CCBB44",
}

MOL_TO_UMOL = 1e6
YEARS_PER_DEC = 10.0


# ======================================================================
# Section 1 — Data loading
# ======================================================================

def load_recon_csv(csv_path: str, col: str = "anom_recon_median") -> pd.Series:
    """Load reconstruction CSV, return specified column as monthly Series."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"[load recon] {os.path.basename(csv_path)} -> {df.shape}")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    return df[col]


def load_cmip6_files(
    model_dirs: List[str], var_name: str,
    time_range: Tuple[int, int],
) -> Dict[str, pd.Series]:
    """Load all CMIP6 model runs, return {name: series}."""
    ts = slice(f"{time_range[0]}-01-01", f"{time_range[1]}-12-31")
    files = sorted(set(
        f for d in model_dirs
        for f in glob.glob(os.path.join(d, "**", "*.nc"), recursive=True)
    ))
    print(f"[scan] {len(files)} NetCDF files")

    results = {}
    for fp in files:
        name, series = _load_single(fp, var_name, ts)
        if name and series is not None:
            results[name] = series
    print(f"[load cmip6] {len(results)} model(s) loaded")
    return results


def _load_single(fp: str, var_name: str,
                 ts: slice) -> Tuple[Optional[str], Optional[pd.Series]]:
    """Load one model file, return (name, monthly global-mean series)."""
    name = os.path.basename(fp).replace(".nc", "")
    try:
        with xr.open_dataset(fp, chunks=None, decode_times=False) as ds:
            ds = xr.decode_cf(ds, use_cftime=True)
            if isinstance(ds.time.to_index(),
                          xr.coding.cftimeindex.CFTimeIndex):
                ds["time"] = ds.indexes["time"].to_datetimeindex(unsafe=True)
            ds = ds.sel(time=ts)
            if ds.time.size == 0:
                return None, None
            da = ds[var_name]
            depth_dim = next((d for d in ("lev", "olevel", "depth")
                              if d in da.dims), None)
            if depth_dim:
                da = da.isel({depth_dim: 0})
            for c in ("lev", "olevel", "depth"):
                if c in da.coords:
                    da = da.drop_vars(c, errors="ignore")
            lat = next((l for l in ("lat", "latitude") if l in da.dims),
                       None)
            lon = next((l for l in ("lon", "longitude") if l in da.dims),
                       None)
            if lat is None:
                return None, None
            w = np.cos(np.deg2rad(da[lat]))
            w.name = "weights"
            mean_da = da.weighted(w).mean(
                [d for d in (lat, lon) if d]
            ).load()
            series = mean_da.to_series()
            series.name = name
            print(f"  [ok] {name} ({len(series)} months)")
            return name, series
    except Exception as e:
        print(f"  [err] {name}: {e}")
        return None, None


# ======================================================================
# Section 2 — Ensemble stats
# ======================================================================

def build_cmip6_ensemble(
    model_dict: Dict[str, pd.Series],
    ref_period: Tuple[int, int],
) -> pd.DataFrame:
    """Align models, compute anomaly, return MMM/STD/MMM_lo/MMM_hi."""
    df = pd.DataFrame(model_dict).sort_index()
    ref_mask = ((df.index.year >= ref_period[0]) &
                (df.index.year <= ref_period[1]))
    clim = df[ref_mask].groupby(df[ref_mask].index.month).mean()
    anom = df.copy()
    for m in range(1, 13):
        mask = df.index.month == m
        anom.loc[mask] = df.loc[mask].values - clim.loc[m].values

    anom["MMM"] = anom.mean(axis=1)
    anom["STD"] = anom.std(axis=1)
    anom["MMM_lo"] = anom["MMM"] - anom["STD"]
    anom["MMM_hi"] = anom["MMM"] + anom["STD"]
    return anom


# ======================================================================
# Section 3 — Bias statistics
# ======================================================================

def compute_bias_stats(recon: pd.Series, mmm: pd.Series,
                       periods: List[Tuple[int, int]]) -> pd.DataFrame:
    """Compute MAD, RMSE, bias, correlation by period."""
    r_a, m_a = recon.align(mmm, join="inner")
    r_yr = r_a.resample("YE").mean()
    m_yr = m_a.resample("YE").mean()

    records = []
    for y0, y1 in periods:
        r = r_yr.loc[str(y0):str(y1)].dropna()
        m = m_yr.loc[str(y0):str(y1)].dropna()
        r, m = r.align(m, join="inner")
        if len(r) < 3:
            continue
        diff = r - m
        slope, _, _, p, _ = stats.linregress(m.values, r.values)
        records.append({
            "period":            f"{y0}–{y1}",
            "n_years":           len(r),
            "MAD (mol/m³)":      round(diff.abs().mean(), 6),
            "RMSE (mol/m³)":     round(float(np.sqrt((diff**2).mean())), 6),
            "mean bias":         round(float(diff.mean()), 6),
            "correlation (r)":   round(float(r.corr(m)), 3),
            "slope (recon/MMM)": round(slope, 3),
            "p":                 round(p, 4),
        })
    return pd.DataFrame(records)


# ======================================================================
# Section 4 — Style
# ======================================================================

def _apply_style(ax, ylabel="", title="", annual=False,
                 show_legend=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(axis="both", direction="in", length=3, width=0.6,
                   labelsize=6)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4,
                  color="#AAAAAA", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=4)
    ax.set_title(title, fontsize=8, pad=4, loc="left", fontweight="bold")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    if annual:
        ax.set_xlabel("Year", fontsize=7, labelpad=3)
        ax.xaxis.set_major_locator(
            mticker.MaxNLocator(integer=True, nbins=8)
        )
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if show_legend:
        leg = ax.legend(fontsize=6, frameon=True, framealpha=0.85,
                        edgecolor="#CCCCCC", borderpad=0.4,
                        labelspacing=0.3, handlelength=1.8, loc="best")
        leg.get_frame().set_linewidth(0.4)


# ======================================================================
# Section 5 — Plot functions
# ======================================================================

def plot_cmip6_spread(cmip6_anom: pd.DataFrame,
                      model_names: List[str],
                      out: str = None, dpi: int = 300) -> plt.Figure:
    """Panel A: individual CMIP6 models + MMM ± 1σ."""
    annual = cmip6_anom.resample("YE").mean()
    x = np.array(annual.index.year)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    for col in model_names:
        if col in annual.columns:
            ax.plot(x, annual[col].values, color=_C["models"],
                    lw=0.5, alpha=0.6, zorder=1)

    ax.fill_between(x, annual["MMM_lo"].values, annual["MMM_hi"].values,
                    color=_C["spread"], alpha=0.35,
                    label="MMM ±1σ (inter-model spread)", zorder=2)
    ax.plot(x, annual["MMM"].values, color=_C["mmm"], lw=1.8,
            label="CMIP6 MMM", zorder=3)
    _apply_style(ax, ylabel="DO anomaly (mol m⁻³)",
                 title="(A) CMIP6 inter-model spread (annual)", annual=True)
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


def plot_mmm_vs_recon(cmip6_anom: pd.DataFrame, recon_wod: pd.Series,
                      recon_iap: Optional[pd.Series],
                      out: str = None, dpi: int = 300) -> plt.Figure:
    """Panel B: MMM ± 1σ vs WOD & IAP (monthly + annual)."""
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4))

    for pi, annual in enumerate([False, True]):
        ax = axes[pi]

        if annual:
            cp = cmip6_anom.resample("YE").mean()
            xc = np.array(cp.index.year)
            wp = recon_wod.resample("YE").mean()
            xw = np.array(wp.index.year)
            ip = recon_iap.resample("YE").mean() if recon_iap is not None else None
            xi = np.array(ip.index.year) if ip is not None else None
        else:
            cp = cmip6_anom
            xc = np.array(cp.index, dtype="datetime64[ms]")
            wp = recon_wod
            xw = np.array(wp.index, dtype="datetime64[ms]")
            ip = recon_iap if recon_iap is not None else None
            xi = np.array(ip.index, dtype="datetime64[ms]") if ip is not None else None

        ax.fill_between(xc, cp["MMM_lo"].values, cp["MMM_hi"].values,
                        color=_C["spread"], alpha=0.30,
                        label="CMIP6 MMM ±1σ", zorder=1)
        mask_m = ~np.isnan(cp["MMM"].values)
        ax.plot(xc[mask_m], cp["MMM"].values[mask_m],
                color=_C["mmm"], lw=1.5, ls="--", label="CMIP6 MMM", zorder=3)
        mask_w = wp.notna().values
        ax.plot(xw[mask_w], wp.values[mask_w], color=_C["wod"], lw=1.8,
                label="WOD reconstruction", zorder=4)
        if ip is not None:
            mask_i = ip.notna().values
            ax.plot(xi[mask_i], ip.values[mask_i], color=_C["iap"], lw=1.8,
                    label="IAP reconstruction", zorder=4)

        if annual:
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin
            ax.axvspan(1900, 1960, alpha=0.08, color="#FF8800", zorder=0)
            ax.text(1930, ymax - span * 0.05, "Early period (sparse obs.)",
                    ha="center", va="top", fontsize=5.5, color="#AA5500",
                    style="italic")
            ax.axvline(1945, color="#AA3377", lw=0.8, ls=":", alpha=0.8)
            ax.text(1945.5, ymax - span * 0.12, "1940s positive anomaly",
                    ha="left", va="top", fontsize=5, color="#AA3377")
            ax.axvline(1960, color="#555555", lw=0.6, ls="--", alpha=0.5)

        _apply_style(ax, ylabel="DO anomaly (mol m⁻³)",
                     title=(f"(B) CMIP6 MMM vs. reconstructions — "
                            f"{'annual' if annual else 'monthly'} mean"),
                     annual=annual)

    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


def plot_bias_timeseries(cmip6_anom: pd.DataFrame, recon_wod: pd.Series,
                         recon_iap: Optional[pd.Series],
                         out: str = None, dpi: int = 300) -> plt.Figure:
    """Panel C: annual bias (recon − MMM) time-series."""
    mmm_yr = cmip6_anom["MMM"].resample("YE").mean()
    wod_yr = recon_wod.resample("YE").mean()
    m_a, w_a = mmm_yr.align(wod_yr, join="inner")
    diff_wod = (w_a - m_a) * 1_000_000

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x_diff = np.array(diff_wod.index.year)
    ax.bar(x_diff, diff_wod.values,
           color=np.where(diff_wod.values >= 0, _C["wod"], _C["iap"]),
           alpha=0.7, width=0.8, zorder=2,
           label="WOD recon − CMIP6 MMM")

    if recon_iap is not None:
        iap_yr = recon_iap.resample("YE").mean()
        m_b, i_b = mmm_yr.align(iap_yr, join="inner")
        diff_iap = (i_b - m_b) * 1_000_000
        ax.plot(np.array(diff_iap.index.year), diff_iap.values,
                color=_C["iap"], lw=1.2, ls="-.", zorder=3,
                label="IAP recon − CMIP6 MMM")

    ax.axhline(0, color="#333333", lw=0.8, zorder=1)
    ax.axvline(1960, color="#555555", lw=0.6, ls="--", alpha=0.5)
    _apply_style(ax, ylabel="Bias (μmol m⁻³)\nrecon − MMM",
                 title="", annual=True)
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


def plot_spread_vs_recon_spread(cmip6_anom: pd.DataFrame,
                                recon_wod_ci_width: pd.Series,
                                out: str = None,
                                dpi: int = 300) -> plt.Figure:
    """Panel D: CMIP6 inter-model spread vs recon uncertainty width."""
    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    cmip6_std = cmip6_anom["STD"].resample("YE").mean()
    ci_ann = recon_wod_ci_width.resample("YE").mean()
    c_a, r_a = cmip6_std.align(ci_ann, join="inner")
    x = c_a.index.year.values

    c_2sig = c_a.values * 2
    r_vals = r_a.values

    ax.plot(x, c_2sig, color=_C["mmm"], lw=1.5, ls="--",
            label="CMIP6 inter-model spread (2σ)")
    ax.plot(x, r_vals, color=_C["wod"], lw=1.5,
            label="WOD reconstruction 95% CI width")

    # Detect crossover year
    diff = r_vals - c_2sig
    cross_year = None
    for i in range(len(diff) - 1):
        if diff[i] > 0 >= diff[i + 1]:
            frac = diff[i] / (diff[i] - diff[i + 1])
            cross_year = int(round(x[i] + frac))
            break

    if cross_year is not None:
        ax.axvline(cross_year, color="#AA3377", lw=1.0, ls=":", alpha=0.9)
        ax.text(cross_year + 0.8, ax.get_ylim()[1] * 0.92,
                f"{cross_year}\nRecon CI < CMIP6 spread",
                fontsize=5, color="#AA3377", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#AA3377", lw=0.4, alpha=0.85))
        print(f"[fig D] Crossover: {cross_year}")

    ax.axvline(1960, color="#555555", lw=0.6, ls="--", alpha=0.5)
    ax.set_ylim(bottom=0)
    _apply_style(ax,
                 ylabel="Uncertainty width (mol m⁻³)",
                 title="(D) CMIP6 spread vs. reconstruction uncertainty",
                 annual=True)
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[save] {out}")
    else:
        plt.show()
    return fig


# ======================================================================
# Section 6 — Rate comparison (panel E)
# ======================================================================

def compute_model_rates(model_dict: Dict[str, pd.Series],
                        cmip6_anom: pd.DataFrame,
                        periods: List[Tuple[int, int]],
                        ref_period: Tuple[int, int],
                        min_years: int = 5) -> pd.DataFrame:
    """Compute deoxygenation rate for each CMIP6 member + MMM."""
    def _ols_rate(y, v):
        s, _, r, p, _ = stats.linregress(y, v)
        return s * MOL_TO_UMOL * YEARS_PER_DEC, r, p

    records = []
    for name, raw in model_dict.items():
        ref = ((raw.index.year >= ref_period[0]) &
               (raw.index.year <= ref_period[1]))
        clim = raw[ref].groupby(raw[ref].index.month).mean()
        anom = raw.copy()
        for m in range(1, 13):
            mask = raw.index.month == m
            anom.loc[mask] = raw.loc[mask].values - clim.loc[m]
        annual = anom.resample("YE").mean()
        family = name.split("_")[0]
        for y0, y1 in periods:
            sub = annual.loc[str(y0):str(y1)].dropna()
            if len(sub) < min_years:
                continue
            rate, r, p = _ols_rate(sub.index.year.astype(float), sub.values)
            records.append({
                "source": name, "family": family,
                "type": "CMIP6 member",
                "period": f"{y0}–{y1}",
                "y0": y0, "y1": y1,
                "n_years": len(sub),
                "rate": round(rate, 4),
                "r": round(r, 3), "p": round(p, 4),
            })

    mmm_ann = cmip6_anom["MMM"].resample("YE").mean()
    for y0, y1 in periods:
        sub = mmm_ann.loc[str(y0):str(y1)].dropna()
        if len(sub) < min_years:
            continue
        rate, r, p = _ols_rate(sub.index.year.astype(float), sub.values)
        records.append({
            "source": "CMIP6 MMM", "family": "MMM",
            "type": "CMIP6 MMM",
            "period": f"{y0}–{y1}",
            "y0": y0, "y1": y1,
            "n_years": len(sub),
            "rate": round(rate, 4),
            "r": round(r, 3), "p": round(p, 4),
        })
    return pd.DataFrame(records)


def plot_rate_comparison_full(
    model_rates: pd.DataFrame,
    recon_rates_wod: pd.DataFrame,
    recon_rates_iap: pd.DataFrame,
    periods_to_plot: Optional[List[str]] = None,
    out: str = None, dpi: int = 300,
) -> plt.Figure:
    """
    Panel E: deoxygenation rate comparison.

    Each subplot shows one period with:
      - CMIP6 member scatter (grey)
      - CMIP6 5–95% range
      - CMIP6 MMM (black diamond)
      - WOD recon ± 95% CI (red)
      - IAP recon ± 95% CI (blue)
    """
    all_periods = model_rates["period"].unique().tolist()
    if periods_to_plot:
        all_periods = [p for p in all_periods if p in periods_to_plot]

    ncols, nrows = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 7.5))

    for pi, period in enumerate(all_periods):
        if pi >= nrows * ncols:
            break
        ax = axes[pi // ncols][pi % ncols]
        mem = model_rates[
            (model_rates["period"] == period) &
            (model_rates["type"] == "CMIP6 member")
        ]["rate"].values
        mmm_row = model_rates[
            (model_rates["period"] == period) &
            (model_rates["type"] == "CMIP6 MMM")
        ]

        if len(mem) > 0:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(mem))
            ax.scatter(np.zeros(len(mem)) + jitter, mem,
                       color="#BBBBBB", s=10, alpha=0.5, zorder=2,
                       label=f"CMIP6 (n={len(mem)})")
            q5, q95 = np.percentile(mem, [5, 95])
            ax.errorbar(0, np.median(mem),
                        yerr=[[np.median(mem) - q5], [q95 - np.median(mem)]],
                        fmt="none", color="#888888", lw=2.0,
                        capsize=6, capthick=1.5, zorder=3)

        if not mmm_row.empty:
            ax.scatter(0, float(mmm_row["rate"].iloc[0]),
                       marker="D", color="#222222", s=55,
                       zorder=5, label="CMIP6 MMM")

        wod_row = recon_rates_wod[recon_rates_wod["period"] == period]
        if not wod_row.empty:
            rv = float(wod_row["rate (μmol/m³/dec)"].iloc[0])
            rl = float(wod_row["CI_lo"].iloc[0])
            rh = float(wod_row["CI_hi"].iloc[0])
            ax.errorbar(1, rv, yerr=[[rv - rl], [rh - rv]],
                        fmt="o", color=_C["wod"], ms=7,
                        elinewidth=1.8, capsize=5, capthick=1.5,
                        zorder=6, label="WOD reconstruction")

        iap_row = recon_rates_iap[recon_rates_iap["period"] == period]
        has_iap = not iap_row.empty
        if has_iap:
            iv = float(iap_row["rate (μmol/m³/dec)"].iloc[0])
            il = float(iap_row["CI_lo"].iloc[0])
            ih = float(iap_row["CI_hi"].iloc[0])
            ax.errorbar(2, iv, yerr=[[iv - il], [ih - iv]],
                        fmt="o", color=_C["iap"], ms=7,
                        elinewidth=1.8, capsize=5, capthick=1.5,
                        zorder=6, label="IAP reconstruction")

        ax.axhline(0, color="#555555", lw=0.7, zorder=1)
        ax.set_xlim(-0.8, 2.8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["CMIP6", "WOD\nrecon", "IAP\nrecon"],
                           fontsize=6)
        ax.tick_params(axis="x", length=0)

        if len(mem) > 0:
            spread = q95 - q5
            ax.text(0, q5 - abs(spread) * 0.08,
                    f"90% spread={spread:.1f}",
                    ha="center", va="top", fontsize=5, color="#666666")

        if not has_iap:
            ax.text(2, ax.get_ylim()[0], "IAP: N/A (starts 1940)",
                    ha="center", va="bottom", fontsize=5,
                    color=_C["iap"], style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=_C["iap"], lw=0.4, alpha=0.8))

        _apply_style(ax, ylabel="Rate (μmol m⁻³ dec⁻¹)",
                     title=f"(E) {period}", annual=False,
                     show_legend=False)

    for idx in range(len(all_periods), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Figure-level legend
    fig.legend(
        handles=[
            mlines.Line2D([], [], marker="o", color="#BBBBBB", ms=5,
                          linestyle="none", label="CMIP6 members"),
            mlines.Line2D([], [], marker="D", color="#222222", ms=6,
                          linestyle="none", label="CMIP6 MMM"),
            mlines.Line2D([], [], marker="o", color=_C["wod"], ms=6,
                          linestyle="none",
                          label="WOD reconstruction (95% CI)"),
            mlines.Line2D([], [], marker="o", color=_C["iap"], ms=6,
                          linestyle="none",
                          label="IAP reconstruction (95% CI)"),
        ],
        loc="lower center", ncol=4, fontsize=6,
        frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
        borderpad=0.6, bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Deoxygenation rate: CMIP6 vs. observational reconstructions",
                 fontsize=9, y=1.01, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
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
    DIR_RES = ("/g12338011ghq/project/hbb/OxygenDiffusion/"
               "Oxy_regression_Surface/results/Results_CSV")
    DIR_OUT = ("/g12338011ghq/project/hbb/OxygenDiffusion/"
               "Oxy_regression_Surface/results/5deg_experiment")
    os.makedirs(DIR_OUT, exist_ok=True)

    CSV_WOD = os.path.join(DIR_RES, "WOD_1900_2014_Ship.csv")
    CSV_IAP = os.path.join(DIR_RES, "IAP_1940_2014_Ship.csv")

    MODEL_DIRS = [
        f"/g12338011ghq/project/hbb/Dataset/CMIP6_DO_GlobalSurface_5deg/{m}"
        for m in ["MPI-ESM1-2-LR", "CanESM5", "MPI-ESM1-2-HR",
                  "CNRM-ESM2-1", "ACCESS-ESM1-5", "MPI-ESM-1-2-HAM"]
    ]

    TIME_RANGE = (1950, 2014)
    VAR_NAME = "o2"
    REF_PERIOD = (1955, 2017)
    PERIODS = [(1950, 2014), (1960, 2014), (1970, 2014),
               (1980, 2014), (1990, 2014), (2000, 2014)]

    # Load data
    recon_wod = load_recon_csv(CSV_WOD)
    recon_iap = load_recon_csv(CSV_IAP)
    df_wod_full = pd.read_csv(CSV_WOD, index_col=0, parse_dates=True)
    ci_width = (df_wod_full["anom_recon_hi_smooth"]
                - df_wod_full["anom_recon_lo_smooth"])
    model_dict = load_cmip6_files(MODEL_DIRS, VAR_NAME, TIME_RANGE)
    if not model_dict:
        print("[ERROR] No CMIP6 models loaded.")
        return

    cmip6 = build_cmip6_ensemble(model_dict, REF_PERIOD)
    cmip6 = cmip6.sort_index()
    cmip6 = cmip6[~cmip6.index.duplicated(keep="first")]
    cmip6 = cmip6.loc["1900-01-01":"2014-12-31"]

    # Bias stats
    print("\n--- Bias: WOD vs CMIP6 MMM ---")
    bias_wod = compute_bias_stats(recon_wod, cmip6["MMM"], PERIODS)
    print(bias_wod.to_string(index=False))

    print("\n--- Bias: IAP vs CMIP6 MMM ---")
    bias_iap = compute_bias_stats(recon_iap, cmip6["MMM"], PERIODS)
    print(bias_iap.to_string(index=False))

    # Rate comparison data
    RATE_CSV = os.path.join(DIR_RES, "deoxygenation_rates_main.csv")
    df_rates = pd.read_csv(RATE_CSV)
    r_wod = df_rates[df_rates["dataset"] == "WOD"].copy()
    r_iap = df_rates[df_rates["dataset"] == "IAP"].copy()

    model_rates = compute_model_rates(model_dict, cmip6, PERIODS, REF_PERIOD)

    # Plot panels
    RATE_PERIODS = [f"{y0}–{y1}" for y0, y1 in PERIODS[:6]]

    plot_bias_timeseries(cmip6, recon_wod, recon_iap,
                         out=os.path.join(DIR_OUT,
                                          "fig7_C_bias_timeseries.png"),
                         dpi=300)

    print("\n[done]")
    plt.show()


if __name__ == "__main__":
    main()
