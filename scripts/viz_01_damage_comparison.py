"""
viz_01_damage_comparison.py
---------------------------
Figure 1: Original (damaged) vs Counterfactual with annotated damage zones.

For each ecosystem × model pair, produces a publication-quality panel showing:
  - Left:  Actual post-hurricane NDVI image
  - Middle: Counterfactual NDVI (what should have been)
  - Right:  ΔNDVI damage map with labeled damage severity zones

Damage zones are computed from ΔNDVI and annotated with:
  - % area damaged
  - Mean ΔNDVI in each zone
  - Zone severity label (Severe / Moderate / Mild)

Usage:
    python scripts/viz_01_damage_comparison.py
    python scripts/viz_01_damage_comparison.py --model eco_transformer
    python scripts/viz_01_damage_comparison.py --quarter 0  # first post-event quarter
    python scripts/viz_01_damage_comparison.py --all-models
"""

import sys
import os
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_MODELS = ["convlstm", "unet_temporal", "eco_transformer", "unet_mamba", "unet_patchtst"]
MODEL_LABELS = {
    "convlstm":       "ConvLSTM",
    "unet_temporal":  "UNet-Temporal",
    "eco_transformer":"EcoTransformer",
    "unet_mamba":     "UNet-Mamba",
    "unet_patchtst":  "UNet-PatchTST",
}

# Damage severity thresholds (ΔNDVI)
DAMAGE_ZONES = [
    ("Mild",     0.05, 0.10, "#ffd700"),
    ("Moderate", 0.10, 0.20, "#ff8c00"),
    ("Severe",   0.20, 1.00, "#cc0000"),
]


def load_arrays(processed_dir: Path, eco: str, model: str) -> Optional[Dict]:
    """Load CF/actual/delta arrays for a given ecosystem + model."""
    base = processed_dir / eco / model
    needed = ["counterfactual.npy", "actual_post.npy", "delta.npy"]
    if not all((base / f).exists() for f in needed):
        logger.warning(f"  Missing arrays for {eco}/{model} at {base}")
        return None
    return {
        "cf":     np.load(str(base / "counterfactual.npy")),
        "actual": np.load(str(base / "actual_post.npy")),
        "delta":  np.load(str(base / "delta.npy")),
    }


def compute_damage_stats(delta_ndvi: np.ndarray, pixel_area_ha: float = 0.01) -> Dict:
    """Compute per-zone damage statistics from a (H, W) ΔNDVI frame."""
    total_pixels = np.sum(np.isfinite(delta_ndvi))
    stats = {}
    for name, lo, hi, _ in DAMAGE_ZONES:
        mask = np.isfinite(delta_ndvi) & (delta_ndvi >= lo) & (delta_ndvi < hi)
        n = int(np.sum(mask))
        mean_d = float(np.mean(delta_ndvi[mask])) if n > 0 else 0.0
        pct = 100.0 * n / max(total_pixels, 1)
        stats[name] = {
            "n_pixels": n,
            "area_ha":  n * pixel_area_ha,
            "mean_delta": mean_d,
            "pct_area":  pct,
        }
    total_impacted = sum(v["n_pixels"] for v in stats.values())
    stats["_total_impacted_pct"] = 100.0 * total_impacted / max(total_pixels, 1)
    stats["_total_impacted_ha"] = total_impacted * pixel_area_ha
    return stats


def plot_damage_comparison(
    arrays: Dict[str, np.ndarray],
    config: Dict,
    eco: str,
    model: str,
    quarter: int,
    out_path: str,
    pixel_size_m: float = 10.0,
) -> None:
    """Create the 3-panel damage comparison figure."""
    eco_cfg = config["ecosystems"][eco]
    ndvi_idx = config["bands"]["indices"]["ndvi"]
    pixel_area_ha = (pixel_size_m ** 2) / 10_000.0

    actual_ndvi = arrays["actual"][quarter, ndvi_idx]   # (H, W)
    cf_ndvi     = arrays["cf"][quarter, ndvi_idx]
    delta_ndvi  = arrays["delta"][quarter, ndvi_idx]

    # Only show positive delta (CF > actual = vegetation loss)
    delta_pos = np.where(delta_ndvi > 0, delta_ndvi, 0.0)
    stats = compute_damage_stats(delta_pos, pixel_area_ha)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor("#0d1117")

    gs = plt.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.08],
                      wspace=0.04, left=0.04, right=0.96, top=0.85, bottom=0.08)

    ax_actual = fig.add_subplot(gs[0, 0])
    ax_cf     = fig.add_subplot(gs[0, 1])
    ax_delta  = fig.add_subplot(gs[0, 2])
    ax_cbar   = fig.add_subplot(gs[0, 3])

    ndvi_kw = dict(cmap="RdYlGn", vmin=0.0, vmax=0.65, aspect="auto",
                   interpolation="nearest")

    # Panel 1: Actual
    im_actual = ax_actual.imshow(actual_ndvi, **ndvi_kw)
    ax_actual.set_title("Actual (Post-Hurricane)", color="white", fontsize=11,
                         fontweight="bold", pad=8)
    ax_actual.axis("off")
    _add_panel_label(ax_actual, "A", color="white")

    # Panel 2: Counterfactual
    ax_cf.imshow(cf_ndvi, **ndvi_kw)
    ax_cf.set_title("Counterfactual (No Hurricane)", color="#7ecfff", fontsize=11,
                     fontweight="bold", pad=8)
    ax_cf.axis("off")
    _add_panel_label(ax_cf, "B", color="#7ecfff")

    # Panel 3: Delta map with damage zone overlays
    damage_cmap = mcolors.LinearSegmentedColormap.from_list(
        "damage", ["#ffffff", "#ffd700", "#ff4500", "#8b0000"], N=256
    )
    delta_display = np.where(delta_pos > 0.01, delta_pos, np.nan)
    ax_delta.imshow(cf_ndvi, cmap="Greys_r", vmin=0, vmax=1, alpha=0.35,
                    aspect="auto", interpolation="nearest")
    im_delta = ax_delta.imshow(delta_display, cmap=damage_cmap, vmin=0.05, vmax=0.35,
                                alpha=0.85, aspect="auto", interpolation="nearest")

    # Annotate damage zones with arrows
    H, W = delta_pos.shape
    for i, (name, lo, hi, color) in enumerate(DAMAGE_ZONES):
        zone_mask = (delta_pos >= lo) & (delta_pos < hi) & np.isfinite(delta_pos)
        if not np.any(zone_mask):
            continue
        ys, xs = np.where(zone_mask)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        s = stats[name]
        label = (f"{name}\n"
                 f"ΔNDVI≥{lo:.2f}\n"
                 f"{s['pct_area']:.1f}% area\n"
                 f"{s['area_ha']:.0f} ha")
        # Offset text outside the image for clarity
        tx = W * 1.02
        ty = H * (0.15 + i * 0.28)
        ax_delta.annotate(
            label,
            xy=(cx, cy), xytext=(tx, ty),
            xycoords="data", textcoords="data",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                            connectionstyle="arc3,rad=0.1"),
            fontsize=8, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                      edgecolor=color, alpha=0.9),
            clip_on=False,
        )

    ax_delta.set_title(
        f"Damage Map (ΔNDVI = CF − Actual)\n"
        f"Total impacted: {stats['_total_impacted_pct']:.1f}% "
        f"({stats['_total_impacted_ha']:.0f} ha)",
        color="#ff6b6b", fontsize=10, fontweight="bold", pad=8
    )
    ax_delta.axis("off")
    _add_panel_label(ax_delta, "C", color="#ff6b6b")

    # Shared NDVI colorbar
    plt.colorbar(im_actual, cax=ax_cbar)
    ax_cbar.yaxis.set_tick_params(color="white", labelcolor="white")
    ax_cbar.set_ylabel("NDVI", color="white", fontsize=9)
    ax_cbar.yaxis.label.set_color("white")

    # Global title
    hurricane = eco_cfg["hurricane"]
    year = eco_cfg["hurricane_year"]
    title = (f"EcoRewind — {eco.title()} Wetland | Hurricane {hurricane} ({year})\n"
             f"Model: {MODEL_LABELS.get(model, model)} | "
             f"Post-event Quarter {quarter + 1}")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=0.98)

    # Style all axes backgrounds
    for ax in [ax_actual, ax_cf, ax_delta]:
        ax.set_facecolor("#0d1117")

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info(f"  Saved: {out_path}")


def _add_panel_label(ax, label: str, color: str = "white"):
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            color=color, fontsize=14, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="black", alpha=0.7))


def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    out_base = Path(config["outputs"]["maps"]) / "publication" / "fig1_damage_comparison"
    out_base.mkdir(parents=True, exist_ok=True)

    ecosystems = [args.ecosystem] if args.ecosystem else list(config["ecosystems"].keys())

    if args.all_models:
        models = [m for m in ALL_MODELS
                  if any((processed_dir / eco / m / "delta.npy").exists()
                         for eco in ecosystems)]
    else:
        models = [args.model or config["model"]["architecture"]]

    logger.info(f"Generating damage comparison figures...")
    logger.info(f"  Ecosystems: {ecosystems}")
    logger.info(f"  Models:     {models}")

    for eco in ecosystems:
        for model in models:
            arrays = load_arrays(processed_dir, eco, model)
            if arrays is None:
                continue
            n_quarters = arrays["delta"].shape[0]
            quarter = min(args.quarter, n_quarters - 1)
            out_path = str(out_base / f"{eco}_{model}_Q{quarter+1:02d}_damage.png")
            logger.info(f"  {eco} / {model} / Q{quarter+1}...")
            try:
                plot_damage_comparison(arrays, config, eco, model, quarter,
                                       out_path, pixel_size_m=10.0)
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

    logger.info(f"\nFigures saved to: {out_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Damage comparison figure")
    parser.add_argument("--model",     type=str, default=None, choices=ALL_MODELS)
    parser.add_argument("--ecosystem", type=str, default=None,
                        choices=["everglades", "mississippi"])
    parser.add_argument("--quarter",   type=int, default=0,
                        help="Post-event quarter index (0 = immediately after event)")
    parser.add_argument("--all-models", action="store_true", dest="all_models")
    main(parser.parse_args())