"""
viz_02_recovery_trajectories.py
--------------------------------
Figure 2: Multi-model NDVI recovery trajectory comparison.

For each ecosystem, plots all models' actual vs counterfactual NDVI curves
on the same axes so the reader can compare which model estimates the most/least
damage and recovery speed.

Panels:
  - Main: NDVI over time (pre-event baseline + post-event actual + CF per model)
  - Inset: ΔNDVI (damage gap) zoomed into post-event window
  - Shaded 95% CI from MC Dropout uncertainty (if available)
  - Vertical dashed line at hurricane event
  - Annotation: estimated recovery quarter per model

Usage:
    python scripts/viz_02_recovery_trajectories.py
    python scripts/viz_02_recovery_trajectories.py --ecosystem everglades
    python scripts/viz_02_recovery_trajectories.py --all-models
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
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_MODELS = ["convlstm", "unet_temporal", "eco_transformer", "unet_mamba", "unet_patchtst"]
MODEL_LABELS = {
    "convlstm":        "ConvLSTM",
    "unet_temporal":   "UNet-Temporal",
    "eco_transformer": "EcoTransformer (ours)",
    "unet_mamba":      "UNet-Mamba",
    "unet_patchtst":   "UNet-PatchTST",
}
# Distinctive color per model
MODEL_COLORS = {
    "convlstm":        "#e74c3c",
    "unet_temporal":   "#3498db",
    "eco_transformer": "#2ecc71",
    "unet_mamba":      "#f39c12",
    "unet_patchtst":   "#9b59b6",
}


def load_metrics(metrics_dir: Path, eco: str, model: str) -> dict:
    """Load pre-computed metrics JSON for a given eco/model."""
    paths = [
        metrics_dir / f"{eco}_{model}_metrics_final.json",
        metrics_dir / f"{eco}_{model}_metrics.json",
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


def load_uncertainty(processed_dir: Path, eco: str, model: str):
    """Load MC Dropout uncertainty arrays if available."""
    p = processed_dir / eco / model / "uncertainty.npy"
    if p.exists():
        return np.load(str(p))
    return None


def ndvi_series_from_arrays(processed_dir: Path, eco: str, model: str, ndvi_idx: int):
    """Compute mean NDVI timeseries from raw arrays."""
    base = processed_dir / eco / model
    cf_path = base / "counterfactual.npy"
    actual_path = base / "actual_post.npy"
    if not cf_path.exists() or not actual_path.exists():
        return None, None
    cf = np.load(str(cf_path))
    actual = np.load(str(actual_path))
    cf_series = [float(np.nanmean(cf[t, ndvi_idx])) for t in range(cf.shape[0])]
    actual_series = [float(np.nanmean(actual[t, ndvi_idx])) for t in range(actual.shape[0])]
    return cf_series, actual_series


def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    metrics_dir   = Path(config["outputs"]["metrics"])
    out_base = Path(config["outputs"]["maps"]) / "publication" / "fig2_trajectories"
    out_base.mkdir(parents=True, exist_ok=True)

    ecosystems = [args.ecosystem] if args.ecosystem else list(config["ecosystems"].keys())

    if args.all_models:
        models_to_plot = [m for m in ALL_MODELS
                          if any((processed_dir / eco / m / "delta.npy").exists()
                                 for eco in ecosystems)]
    else:
        models_to_plot = [args.model] if args.model else [
            m for m in ALL_MODELS
            if any((processed_dir / eco / m / "counterfactual.npy").exists()
                   for eco in ecosystems)
        ]

    ndvi_idx = config["bands"]["indices"]["ndvi"]
    logger.info(f"Models to plot: {models_to_plot}")

    for eco in ecosystems:
        eco_cfg = config["ecosystems"][eco]
        hurricane = eco_cfg["hurricane"]
        hurricane_year = eco_cfg["hurricane_year"]
        t_event = eco_cfg["t_event"]

        logger.info(f"\n--- {eco.upper()} ---")

        # Collect per-model data
        model_data = {}
        for model in models_to_plot:
            cf_s, actual_s = ndvi_series_from_arrays(processed_dir, eco, model, ndvi_idx)
            if cf_s is None:
                continue
            metrics = load_metrics(metrics_dir, eco, model)
            unc = load_uncertainty(processed_dir, eco, model)
            unc_series = None
            if unc is not None:
                unc_series = [float(np.nanmean(unc[t, ndvi_idx])) for t in range(unc.shape[0])]
            model_data[model] = {
                "cf": cf_s, "actual": actual_s,
                "ttc": metrics.get("time_to_convergence_quarters"),
                "uncertainty": unc_series,
                "gap_closing": metrics.get("gap_closing_rate", float("nan")),
                "delta_q1": metrics.get("delta_ndvi_mean_timeseries", [0])[0]
                            if metrics.get("delta_ndvi_mean_timeseries") else 0,
            }

        if not model_data:
            logger.warning(f"  No data for {eco} — skipping")
            continue

        n_post = max(len(v["actual"]) for v in model_data.values())
        quarters = np.arange(n_post)

        # ── Figure setup ────────────────────────────────────────────────────
        fig, (ax_main, ax_gap) = plt.subplots(
            2, 1, figsize=(14, 9),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
        )
        fig.patch.set_facecolor("#fafafa")

        for ax in [ax_main, ax_gap]:
            ax.set_facecolor("#f5f5f5")
            for spine in ax.spines.values():
                spine.set_color("#cccccc")

        # ── Plot each model ──────────────────────────────────────────────────
        for model, data in model_data.items():
            color = MODEL_COLORS.get(model, "#888888")
            label = MODEL_LABELS.get(model, model)
            n = len(data["actual"])
            qs = quarters[:n]

            # Actual trajectory (solid)
            ax_main.plot(qs, data["actual"], "-o", color=color,
                         linewidth=2, markersize=4, label=f"{label} — Actual", zorder=5)

            # CF trajectory (dashed, lighter)
            ax_main.plot(qs, data["cf"], "--", color=color,
                         linewidth=1.5, alpha=0.6, label=f"{label} — CF", zorder=4)

            # MC uncertainty band
            if data["uncertainty"]:
                unc = np.array(data["uncertainty"][:n])
                cf  = np.array(data["cf"][:n])
                ax_main.fill_between(qs, cf - 1.96*unc, cf + 1.96*unc,
                                     alpha=0.08, color=color)

            # Gap (damage)
            gap = [c - a for c, a in zip(data["cf"][:n], data["actual"][:n])]
            ax_gap.fill_between(qs, 0, gap, alpha=0.35, color=color)
            ax_gap.plot(qs, gap, "-", color=color, linewidth=1.5, zorder=5)

            # Recovery annotation
            ttc = data["ttc"]
            if ttc is not None and ttc < n:
                ax_main.axvline(ttc, color=color, linestyle=":", linewidth=1, alpha=0.7)
                ax_main.text(ttc + 0.1, min(data["actual"]) + 0.02,
                             f"Recovery\nQ{ttc+1}", color=color, fontsize=7.5,
                             ha="left", va="bottom",
                             bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                       ec=color, alpha=0.85))

        # Hurricane event line
        ax_main.axvline(-0.5, color="#cc0000", linestyle="--", linewidth=2, zorder=10)
        ax_main.text(-0.4, ax_main.get_ylim()[1] * 0.97 if ax_main.get_ylim()[1] > 0 else 0.65,
                     f"Hurricane {hurricane}\n({hurricane_year})",
                     color="#cc0000", fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor="#cc0000", alpha=0.9))

        ax_gap.axvline(-0.5, color="#cc0000", linestyle="--", linewidth=2)
        ax_gap.axhline(0, color="#555555", linewidth=0.8)

        # Axes labels
        ax_main.set_ylabel("Mean NDVI", fontsize=12, fontweight="bold")
        ax_main.set_title(
            f"EcoRewind — {eco.title()} Wetland | Hurricane {hurricane} ({hurricane_year})\n"
            f"NDVI Recovery Trajectories: Actual vs Counterfactual",
            fontsize=13, fontweight="bold", pad=12
        )
        ax_main.grid(alpha=0.25, linestyle="--")
        ax_main.set_xticks([])

        ax_gap.set_ylabel("ΔNDVI\n(damage gap)", fontsize=10)
        ax_gap.set_xlabel("Post-event quarters", fontsize=12)
        ax_gap.set_xticks(quarters[::2])
        ax_gap.set_xticklabels([f"Q{int(t)+1}" for t in quarters[::2]], fontsize=9)
        ax_gap.grid(alpha=0.25, linestyle="--")

        # Legend (consolidated)
        handles = []
        for model in model_data:
            color = MODEL_COLORS.get(model, "#888")
            label = MODEL_LABELS.get(model, model)
            handles.append(Line2D([0], [0], color=color, lw=2,
                                  label=f"{label}"))
        handles.append(Line2D([0], [0], color="k", lw=2, linestyle="-", label="Actual"))
        handles.append(Line2D([0], [0], color="k", lw=1.5, linestyle="--",
                               alpha=0.5, label="Counterfactual"))
        ax_main.legend(handles=handles, fontsize=9, loc="lower right",
                       framealpha=0.9, ncol=2)

        out_path = out_base / f"{eco}_recovery_trajectories.png"
        plt.savefig(str(out_path), dpi=180, bbox_inches="tight", facecolor="#fafafa")
        plt.close()
        logger.info(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default=None)
    parser.add_argument("--ecosystem",  type=str, default=None,
                        choices=["everglades", "mississippi"])
    parser.add_argument("--all-models", action="store_true", dest="all_models")
    main(parser.parse_args())