"""
maps.py
-------
Visualization outputs for ECO-REWIND:
 
  1. ΔNDVI / ΔNDWI / ΔSAR_VV damage maps (spatial)
  2. Trajectory comparison plots (actual vs counterfactual over time)
  3. Recovery progress plots
  4. Impacted area and carbon loss bar charts
"""
 
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering for servers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Damage maps
# ---------------------------------------------------------------------------
 
def plot_delta_maps(
    cf_results: Dict[str, np.ndarray],
    config: Dict[str, Any],
    ecosystem: str,
    output_dir: str,
    quarter_index: int = 0,
) -> None:
    """
    Plot ΔNDVI, ΔNDWI, and ΔSAR_VV spatial maps for a given post-event quarter.
 
    quarter_index: which post-event quarter to visualize (0 = first post-event)
    """
    out_dir = Path(output_dir) / ecosystem
    out_dir.mkdir(parents=True, exist_ok=True)
 
    delta = cf_results["delta"]                # (T_post, C, H, W)
    cf = cf_results["counterfactual"]
    actual = cf_results["actual_post"]
 
    if quarter_index >= delta.shape[0]:
        logger.warning(f"quarter_index={quarter_index} out of range, using 0")
        quarter_index = 0
 
    ndvi_idx = config["bands"]["indices"]["ndvi"]
    ndwi_idx = config["bands"]["indices"]["ndwi"]
    sar_idx = config["bands"]["indices"]["sar_vv"]
    eco_cfg = config["ecosystems"][ecosystem.lower()]
 
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"ECO-REWIND: {ecosystem.title()} — Hurricane {eco_cfg['hurricane']} Impact\n"
        f"Post-event Quarter {quarter_index + 1}",
        fontsize=14, fontweight="bold"
    )
 
    # Row 0: Counterfactual (what should have been)
    # Row 1: Actual (what happened)
    band_specs = [
        (ndvi_idx, "NDVI", "RdYlGn", -1, 1),
        (ndwi_idx, "NDWI", "RdYlBu", -1, 1),
        (sar_idx, "SAR_VV (dB)", "plasma", -25, 0),
    ]
 
    for col, (b_idx, name, cmap, vmin, vmax) in enumerate(band_specs):
        cf_band = cf[quarter_index, b_idx]
        act_band = actual[quarter_index, b_idx]
        d_band = delta[quarter_index, b_idx]
 
        # Top row: Counterfactual
        im0 = axes[0, col].imshow(cf_band, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, col].set_title(f"Counterfactual {name}", fontsize=10)
        axes[0, col].axis("off")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
 
        # Bottom row: Actual
        im1 = axes[1, col].imshow(act_band, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axes[1, col].set_title(f"Actual {name}", fontsize=10)
        axes[1, col].axis("off")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)
 
    plt.tight_layout()
    fname = out_dir / f"damage_maps_Q{quarter_index+1:02d}.png"
    plt.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {fname}")
 
    # Also save ΔNDVI-only map (main result figure)
    _plot_delta_ndvi_map(
        delta[:, ndvi_idx], ecosystem, eco_cfg, out_dir
    )
 
 
def _plot_delta_ndvi_map(
    delta_ndvi: np.ndarray,   # (T_post, H, W)
    ecosystem: str,
    eco_cfg: Dict,
    out_dir: Path,
) -> None:
    """High-quality ΔNDVI damage map at peak impact (first post-event quarter)."""
    d0 = delta_ndvi[0]   # first post-event quarter
 
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
 
    # Custom diverging colormap: green = no damage, red = severe damage
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(d0, cmap=cmap, vmin=0, vmax=0.3, aspect="auto")
    plt.colorbar(im, ax=ax, label="ΔNDVI (Counterfactual − Actual)", fraction=0.046)
 
    ax.set_title(
        f"Hurricane {eco_cfg['hurricane']} ({eco_cfg['hurricane_year']}) — "
        f"Vegetation Damage Map\n{ecosystem.title()}",
        fontsize=13, fontweight="bold"
    )
    ax.axis("off")
 
    # Add impact contour at threshold
    threshold = 0.05
    try:
        ax.contour(d0, levels=[threshold], colors=["black"], linewidths=0.5, alpha=0.5)
    except Exception:
        pass
 
    plt.tight_layout()
    fname = out_dir / "delta_ndvi_impact_map.png"
    plt.savefig(str(fname), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {fname}")
 
 
# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------
 
def plot_trajectories(
    metrics_results: Dict[str, Any],
    config: Dict[str, Any],
    ecosystem: str,
    output_dir: str,
) -> None:
    """
    Plot actual vs counterfactual NDVI trajectories over time.
    The key figure for thesis / publication.
    """
    out_dir = Path(output_dir) / ecosystem
    out_dir.mkdir(parents=True, exist_ok=True)
 
    eco_cfg = config["ecosystems"][ecosystem.lower()]
    t_event = eco_cfg["t_event"]
 
    actual_ts = metrics_results.get("actual_ndvi_timeseries", [])
    cf_ts = metrics_results.get("cf_ndvi_timeseries", [])
    gap_ts = metrics_results.get("gap_timeseries", [])
 
    if not actual_ts or not cf_ts:
        logger.warning(f"[{ecosystem}] No trajectory data to plot")
        return
 
    T_post = len(actual_ts)
    quarters = np.arange(T_post)
    quarter_labels = [f"Q{(t_event + t) % 4 + 1} {(t_event + t) // 4 + 2016}"
                      for t in range(T_post)]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"ECO-REWIND: {ecosystem.title()} — Counterfactual vs Actual NDVI\n"
        f"Hurricane {eco_cfg['hurricane']} ({eco_cfg['hurricane_year']})",
        fontsize=13, fontweight="bold"
    )
 
    # Top: trajectory comparison
    ax1.plot(quarters, cf_ts, "g-o", markersize=5, label="Counterfactual (no hurricane)",
             linewidth=2, color="#2ecc71")
    ax1.plot(quarters, actual_ts, "r-s", markersize=5, label="Actual (post-hurricane)",
             linewidth=2, color="#e74c3c")
 
    # Shade the gap
    ax1.fill_between(
        quarters,
        actual_ts, cf_ts,
        where=[c > a for c, a in zip(cf_ts, actual_ts)],
        alpha=0.2, color="red", label="Damage gap"
    )
 
    ax1.set_ylabel("Mean NDVI", fontsize=11)
    ax1.set_ylim(0, max(max(cf_ts, default=0.7) * 1.1, 0.7))
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(quarters[::2])
    ax1.set_xticklabels([quarter_labels[i] for i in range(0, T_post, 2)], rotation=45, fontsize=8)
 
    # Bottom: gap (damage) over time
    ax2.bar(quarters, gap_ts if gap_ts else [0] * T_post,
            color=["#e74c3c" if g > 0 else "#2ecc71" for g in (gap_ts or [0]*T_post)],
            alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Post-event quarter", fontsize=11)
    ax2.set_ylabel("ΔNDVI gap", fontsize=11)
    ax2.set_xticks(quarters[::2])
    ax2.set_xticklabels([quarter_labels[i] for i in range(0, T_post, 2)], rotation=45, fontsize=8)
    ax2.grid(alpha=0.3)
 
    plt.tight_layout()
    fname = out_dir / "trajectory_comparison.png"
    plt.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {fname}")
 
 
def plot_recovery_metrics(
    metrics_results: Dict[str, Any],
    config: Dict[str, Any],
    ecosystem: str,
    output_dir: str,
) -> None:
    """Bar chart: impacted area and carbon loss over time."""
    out_dir = Path(output_dir) / ecosystem
    out_dir.mkdir(parents=True, exist_ok=True)
 
    eco_cfg = config["ecosystems"][ecosystem.lower()]
    impacted_ha = metrics_results.get("impacted_ha_timeseries", [])
    carbon_loss = metrics_results.get("carbon_loss_tco2_eq", 0)
 
    if not impacted_ha:
        logger.warning(f"[{ecosystem}] No impacted_ha timeseries to plot")
        return
 
    T = len(impacted_ha)
    quarters = np.arange(T)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"ECO-REWIND: {ecosystem.title()} — Ecological Damage Metrics\n"
        f"Hurricane {eco_cfg['hurricane']} ({eco_cfg['hurricane_year']})",
        fontsize=13, fontweight="bold"
    )
 
    # Impacted area over time
    ax1.bar(quarters, impacted_ha, color="#e74c3c", alpha=0.8)
    ax1.set_xlabel("Post-event quarter")
    ax1.set_ylabel("Impacted area (hectares)")
    ax1.set_title(f"Vegetation Loss Area (ΔNDVI > {config['inference']['impact_threshold']})")
    ax1.grid(alpha=0.3, axis="y")
 
    # Carbon loss single bar
    ax2.bar(["Estimated\nCarbon Loss"], [carbon_loss], color="#8e44ad", alpha=0.8, width=0.5)
    ax2.set_ylabel("tCO₂-equivalent")
    ax2.set_title("Estimated Carbon Stock Loss")
    ax2.grid(alpha=0.3, axis="y")
 
    # Annotate
    ax2.text(
        0, carbon_loss * 0.5,
        f"{carbon_loss:,.0f} tCO₂-eq",
        ha="center", va="center", fontsize=12, color="white", fontweight="bold"
    )
 
    plt.tight_layout()
    fname = out_dir / "recovery_metrics.png"
    plt.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {fname}")
 
 
# ---------------------------------------------------------------------------
# Ablation comparison
# ---------------------------------------------------------------------------
 
def plot_ablation_comparison(
    ablation_results: Dict[str, Dict],
    output_dir: str,
) -> None:
    """
    Bar chart comparing val loss and NDVI MSE across ablation experiments.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    experiments = list(ablation_results.keys())
    val_losses = [r.get("best_val_loss", float("nan")) for r in ablation_results.values()]
    ndvi_mses = [r.get("ndvi_mse", float("nan")) for r in ablation_results.values()]
 
    x = np.arange(len(experiments))
    width = 0.35
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ECO-REWIND: Ablation Study Results", fontsize=13, fontweight="bold")
 
    bars1 = ax1.bar(x, val_losses, width, color="#3498db", alpha=0.8)
    ax1.set_xlabel("Experiment")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Validation Loss by Ablation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments, rotation=30, ha="right")
    ax1.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars1, val_losses):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)
 
    bars2 = ax2.bar(x, ndvi_mses, width, color="#2ecc71", alpha=0.8)
    ax2.set_xlabel("Experiment")
    ax2.set_ylabel("NDVI MSE")
    ax2.set_title("NDVI Reconstruction MSE by Ablation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiments, rotation=30, ha="right")
    ax2.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars2, ndvi_mses):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)
 
    plt.tight_layout()
    fname = out_dir / "ablation_comparison.png"
    plt.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ablation plot: {fname}")