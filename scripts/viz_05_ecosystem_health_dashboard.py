"""
viz_05_ecosystem_health_dashboard.py
--------------------------------------
Figure 5: Ecosystem health dashboard — a single-page summary figure
suitable for a poster or paper Figure 1.

Layout:
  Top-left:   Study area schematic (text-based map showing ecosystems + hurricane tracks)
  Top-right:  Carbon loss estimates with uncertainty (grouped bar, all models)
  Mid-left:   ΔNDVI distribution histogram: damaged vs undamaged pixels
  Mid-right:  Scatter: predicted NDVI vs actual NDVI (all post-event pixels)
              coloured by damage severity
  Bottom:     Multi-band comparison strip: Blue, NIR, NDVI, SAR_VV
              showing how each band changes actual vs CF

Usage:
    python scripts/viz_05_ecosystem_health_dashboard.py
    python scripts/viz_05_ecosystem_health_dashboard.py --model eco_transformer
    python scripts/viz_05_ecosystem_health_dashboard.py --ecosystem everglades
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_MODELS = ["convlstm", "unet_temporal", "eco_transformer", "unet_mamba", "unet_patchtst"]
MODEL_LABELS = {
    "convlstm":        "ConvLSTM",
    "unet_temporal":   "UNet-Temporal",
    "eco_transformer": "EcoTransformer",
    "unet_mamba":      "UNet-Mamba",
    "unet_patchtst":   "UNet-PatchTST",
}
MODEL_COLORS = {
    "convlstm":        "#e74c3c",
    "unet_temporal":   "#3498db",
    "eco_transformer": "#2ecc71",
    "unet_mamba":      "#f39c12",
    "unet_patchtst":   "#9b59b6",
}


def load_metrics(metrics_dir: Path, eco: str, model: str) -> dict:
    for fname in [f"{eco}_{model}_metrics_final.json", f"{eco}_{model}_metrics.json"]:
        p = metrics_dir / fname
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


def load_arrays(processed_dir: Path, eco: str, model: str):
    base = processed_dir / eco / model
    needed = ["counterfactual.npy", "actual_post.npy", "delta.npy"]
    if not all((base / f).exists() for f in needed):
        return None
    return {
        "cf":     np.load(str(base / "counterfactual.npy")),
        "actual": np.load(str(base / "actual_post.npy")),
        "delta":  np.load(str(base / "delta.npy")),
    }


def plot_damage_histogram(ax, delta_ndvi_flat: np.ndarray, ecosystem: str, hurricane: str):
    """Panel: distribution of ΔNDVI values."""
    valid = delta_ndvi_flat[np.isfinite(delta_ndvi_flat)]
    damaged = valid[valid > 0.05]
    undamaged = valid[valid <= 0.05]

    bins = np.linspace(-0.1, 0.5, 60)
    ax.hist(undamaged, bins=bins, color="#2ecc71", alpha=0.7, label="Undamaged (ΔNDVI≤0.05)",
            density=True, edgecolor="none")
    ax.hist(damaged,   bins=bins, color="#e74c3c", alpha=0.7, label="Damaged (ΔNDVI>0.05)",
            density=True, edgecolor="none")
    ax.axvline(0.05, color="#333", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(0.10, color="#cc7700", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.axvline(0.20, color="#cc0000", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.text(0.06, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 5,
            "Mild", fontsize=7, color="#cc7700")
    ax.set_xlabel("ΔNDVI (Counterfactual − Actual)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f"D  ΔNDVI Distribution — {ecosystem.title()}\nHurricane {hurricane}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    pct = 100 * len(damaged) / max(len(valid), 1)
    ax.text(0.98, 0.70, f"{pct:.1f}% pixels\ndamaged",
            transform=ax.transAxes, ha="right", fontsize=9, color="#e74c3c",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#e74c3c", alpha=0.9))
    ax.grid(alpha=0.3, linestyle="--")


def plot_pred_vs_actual_scatter(ax, cf_ndvi_flat: np.ndarray, actual_ndvi_flat: np.ndarray,
                                 delta_flat: np.ndarray):
    """Panel: predicted (CF) vs actual NDVI scatter."""
    valid = (np.isfinite(cf_ndvi_flat) & np.isfinite(actual_ndvi_flat) &
             np.isfinite(delta_flat))
    if valid.sum() < 100:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        return

    # Subsample for speed
    idx = np.random.choice(np.where(valid)[0], size=min(8000, valid.sum()), replace=False)
    x = actual_ndvi_flat[idx]
    y = cf_ndvi_flat[idx]
    d = delta_flat[idx]
    d_clamp = np.clip(d, 0, 0.3)

    damage_cmap = LinearSegmentedColormap.from_list(
        "dmg", ["#2ecc71", "#ffd700", "#ff4500", "#8b0000"])
    sc = ax.scatter(x, y, c=d_clamp, cmap=damage_cmap, s=3, alpha=0.5,
                    vmin=0, vmax=0.3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="ΔNDVI", fraction=0.035)

    # 1:1 line
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, alpha=0.5, label="1:1")
    ax.set_xlabel("Actual NDVI (post-event)", fontsize=9)
    ax.set_ylabel("Counterfactual NDVI", fontsize=9)
    ax.set_title("E  Predicted vs Actual NDVI\n(colored by damage intensity)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")

    # R² annotation
    from numpy.polynomial import polynomial as P
    corr = np.corrcoef(x, y)[0, 1] ** 2
    ax.text(0.05, 0.92, f"R² = {corr:.3f}", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="#2c3e50")


def plot_multiband_strip(ax_list, arrays: dict, config: dict, quarter: int = 0):
    """Panel: strip showing CF vs actual for multiple bands."""
    bands_to_show = [
        (config["bands"]["indices"].get("blue", 0), "Blue", "Blues"),
        (config["bands"]["indices"].get("nir", 3),  "NIR",  "Greens"),
        (config["bands"]["indices"].get("ndvi", 4), "NDVI", "RdYlGn"),
        (config["bands"]["indices"].get("sar_vv", 6), "SAR_VV", "plasma"),
    ]

    for i, (b_idx, bname, cmap) in enumerate(bands_to_show):
        ax = ax_list[i]
        cf_b  = arrays["cf"][quarter, b_idx]
        act_b = arrays["actual"][quarter, b_idx]
        vmin, vmax = np.nanpercentile(act_b, 2), np.nanpercentile(act_b, 98)

        # Left: CF; Right: Actual side by side
        combined = np.concatenate([cf_b, act_b], axis=1)
        ax.imshow(combined, cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect="auto", interpolation="nearest")
        ax.set_title(f"{bname}\n[CF | Actual]", fontsize=8, pad=2)
        ax.axis("off")
        # Divider line
        ax.axvline(cf_b.shape[1], color="white", linewidth=2)


def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    metrics_dir   = Path(config["outputs"]["metrics"])
    out_base = Path(config["outputs"]["maps"]) / "publication" / "fig5_dashboard"
    out_base.mkdir(parents=True, exist_ok=True)

    ecosystems = [args.ecosystem] if args.ecosystem else list(config["ecosystems"].keys())
    models     = [args.model] if args.model else [
        m for m in ALL_MODELS
        if any((processed_dir / eco / m / "delta.npy").exists() for eco in ecosystems)
    ]

    ndvi_idx = config["bands"]["indices"]["ndvi"]

    for eco in ecosystems:
        eco_cfg   = config["ecosystems"][eco]
        hurricane = eco_cfg["hurricane"]
        hurricane_year = eco_cfg["hurricane_year"]

        # Pick best model (first available)
        selected_model = None
        selected_arrays = None
        for m in ["eco_transformer"] + models:
            a = load_arrays(processed_dir, eco, m)
            if a is not None:
                selected_model = m
                selected_arrays = a
                break

        if selected_arrays is None:
            logger.warning(f"  No arrays for {eco} — skipping dashboard")
            continue

        logger.info(f"  {eco}: using {selected_model} for main panels")

        # ── Figure ────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor("#fafafa")

        # Panel A: Carbon loss bar (top-right)
        ax_carbon = fig.add_axes([0.54, 0.72, 0.42, 0.22])
        ax_carbon.set_facecolor("#f0f0f0")
        model_labels_plot, carbon_vals, err_vals = [], [], []
        for m in models:
            m_metrics = load_metrics(metrics_dir, eco, m)
            if not m_metrics:
                continue
            carbon = m_metrics.get("carbon_loss_tco2_eq", 0)
            model_labels_plot.append(MODEL_LABELS.get(m, m))
            carbon_vals.append(carbon)
            err_vals.append(carbon * 0.12)  # 12% uncertainty proxy

        if carbon_vals:
            colors = [MODEL_COLORS.get(m, "#888") for m in models if
                      load_metrics(metrics_dir, eco, m)]
            bars = ax_carbon.bar(range(len(carbon_vals)), carbon_vals, color=colors,
                                  alpha=0.85, edgecolor="white", yerr=err_vals,
                                  capsize=4, ecolor="#555")
            ax_carbon.set_xticks(range(len(model_labels_plot)))
            ax_carbon.set_xticklabels(model_labels_plot, rotation=25, ha="right", fontsize=8)
            ax_carbon.set_ylabel("tCO₂-equivalent", fontsize=9)
            ax_carbon.set_title(f"A  Carbon Stock Loss — {eco.title()}\nHurricane {hurricane} ({hurricane_year})",
                                 fontsize=9, fontweight="bold")
            ax_carbon.grid(axis="y", alpha=0.3, linestyle="--")
            for bar, val in zip(bars, carbon_vals):
                ax_carbon.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(err_vals) * 0.05,
                               f"{val:,.0f}", ha="center", fontsize=7, rotation=15)

        # Panel B: Study context (text panel)
        ax_context = fig.add_axes([0.03, 0.72, 0.47, 0.22])
        ax_context.set_facecolor("#f8f4e8")
        ax_context.set_xlim(0, 1)
        ax_context.set_ylim(0, 1)
        ax_context.axis("off")

        context_text = (
            f"EcoRewind: Counterfactual Ecosystem Forecasting\n"
            f"{'─' * 55}\n\n"
            f"Ecosystem:    {eco.title()} Wetland\n"
            f"Disturbance:  Hurricane {hurricane} ({hurricane_year})\n"
            f"Sensor:       Sentinel-2 (10m) + Sentinel-1 SAR\n"
            f"Bands:        Blue, Green, Red, NIR, NDVI, NDWI, SAR_VV\n"
            f"Time series:  32 quarters (2016–2023, quarterly composites)\n\n"
            f"Counterfactual: what would the ecosystem have looked like\n"
            f"without the hurricane? Predicted by deep spatiotemporal\n"
            f"learning models trained on pre-disturbance dynamics.\n\n"
            f"Key question: How much vegetation was lost? How fast\n"
            f"is it recovering?"
        )
        ax_context.text(0.04, 0.96, context_text,
                         transform=ax_context.transAxes,
                         va="top", ha="left", fontsize=9,
                         fontfamily="monospace", color="#2c3e50",
                         linespacing=1.6)

        # Panel C: Impacted area per quarter (time series bars)
        ax_impact = fig.add_axes([0.03, 0.48, 0.47, 0.20])
        ax_impact.set_facecolor("#f0f0f0")
        for m in models:
            m_metrics = load_metrics(metrics_dir, eco, m)
            if not m_metrics:
                continue
            ha_ts = m_metrics.get("impacted_ha_timeseries", [])
            if ha_ts:
                ax_impact.plot(range(len(ha_ts)), ha_ts,
                               "o-", color=MODEL_COLORS.get(m, "#888"),
                               linewidth=1.5, markersize=4,
                               label=MODEL_LABELS.get(m, m))
        ax_impact.set_xlabel("Post-event quarter", fontsize=9)
        ax_impact.set_ylabel("Impacted area (ha)", fontsize=9)
        ax_impact.set_title("B  Vegetation Loss Area Over Time", fontsize=9, fontweight="bold")
        ax_impact.legend(fontsize=7, ncol=2)
        ax_impact.grid(alpha=0.3, linestyle="--")

        # Panel D: ΔNDVI histogram
        ax_hist = fig.add_axes([0.54, 0.48, 0.42, 0.20])
        ax_hist.set_facecolor("#f0f0f0")
        delta_flat = selected_arrays["delta"][:, ndvi_idx].ravel()
        plot_damage_histogram(ax_hist, delta_flat, eco, hurricane)

        # Panel E: Scatter pred vs actual
        ax_scatter = fig.add_axes([0.03, 0.24, 0.42, 0.20])
        ax_scatter.set_facecolor("#f0f0f0")
        cf_flat     = selected_arrays["cf"][:, ndvi_idx].ravel()
        actual_flat = selected_arrays["actual"][:, ndvi_idx].ravel()
        plot_pred_vs_actual_scatter(ax_scatter, cf_flat, actual_flat, delta_flat)

        # Panel F: Multi-band strip
        band_strip_axes = []
        for bi in range(4):
            left = 0.54 + bi * 0.115
            ax_b = fig.add_axes([left, 0.24, 0.10, 0.20])
            ax_b.set_facecolor("#1a1a2e")
            band_strip_axes.append(ax_b)
        try:
            plot_multiband_strip(band_strip_axes, selected_arrays, config, quarter=0)
            band_strip_axes[0].set_ylabel("F  Multi-band\nComparison\n[CF | Actual]",
                                          fontsize=8, color="#333", labelpad=3)
        except Exception as e:
            logger.warning(f"  Band strip failed: {e}")

        # Panel G: Summary stats text box
        ax_summary = fig.add_axes([0.03, 0.04, 0.92, 0.16])
        ax_summary.set_facecolor("#1a1a2e")
        ax_summary.axis("off")

        # Build summary row
        summary_parts = []
        for m in models[:5]:
            m_metrics = load_metrics(metrics_dir, eco, m)
            if not m_metrics:
                continue
            carbon = m_metrics.get("carbon_loss_tco2_eq", 0)
            area   = m_metrics.get("impacted_ha_max", 0)
            ttc    = m_metrics.get("time_to_convergence_quarters")
            ttc_s  = f"{ttc}Q" if ttc else ">"
            summary_parts.append(
                f"{MODEL_LABELS.get(m,m)[:14]:<14}  "
                f"Area: {area:>8,.0f} ha  "
                f"Carbon: {carbon:>10,.0f} tCO₂  "
                f"Recovery: {ttc_s}"
            )

        summary_text = (
            f"{'─'*90}\n"
            f"  {'Model':<14}  {'Max Area Impacted':>18}  {'Carbon Loss':>16}  "
            f"{'Recovery Time':>14}\n"
            f"{'─'*90}\n"
        ) + "\n".join(f"  {p}" for p in summary_parts)

        ax_summary.text(0.02, 0.95, summary_text,
                         transform=ax_summary.transAxes,
                         va="top", ha="left", fontsize=8,
                         fontfamily="monospace", color="#ffffff",
                         linespacing=1.5)

        fig.suptitle(
            f"EcoRewind — Ecosystem Health Dashboard\n"
            f"{eco.title()} Wetland | Hurricane {hurricane} ({hurricane_year})",
            fontsize=14, fontweight="bold", color="#1a1a2e", y=0.99
        )

        out_path = out_base / f"{eco}_health_dashboard.png"
        plt.savefig(str(out_path), dpi=160, bbox_inches="tight", facecolor="#fafafa")
        plt.close()
        logger.info(f"  Saved: {out_path}")

    logger.info(f"\nDashboard figures saved to: {out_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str, default=None)
    parser.add_argument("--ecosystem", type=str, default=None,
                        choices=["everglades", "mississippi"])
    main(parser.parse_args())