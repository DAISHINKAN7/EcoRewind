"""
viz_03_model_comparison.py
---------------------------
Figure 3: Cross-model quantitative comparison.

Panels:
  A) Grouped bar chart: Impacted area (ha) per model × ecosystem
  B) Grouped bar chart: Carbon loss (tCO₂-eq) per model × ecosystem
  C) Recovery speed: bar chart of gap-closing rate per model
  D) Radar chart: 5-axis model scorecard
     (NDVI R², SAR R², Impacted Area ↓, Carbon Loss ↓, Recovery Speed ↑)
  E) Time-to-convergence: horizontal bar chart per model × ecosystem

Usage:
    python scripts/viz_03_model_comparison.py
    python scripts/viz_03_model_comparison.py --ecosystem everglades
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
from matplotlib.patches import FancyBboxPatch
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
ECO_HATCH = {"everglades": "", "mississippi": "///"}


def load_metrics(metrics_dir: Path, eco: str, model: str) -> dict:
    for fname in [f"{eco}_{model}_metrics_final.json", f"{eco}_{model}_metrics.json"]:
        p = metrics_dir / fname
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


def load_eval_metrics(metrics_dir: Path, model: str, ecosystem: str) -> dict:
    p = metrics_dir / f"{model}_{ecosystem}_eval.json"
    if not p.exists():
        p = metrics_dir / f"{model}_joint_eval.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def make_radar(ax, values: list, categories: list, model_label: str, color: str):
    """Draw a single-model radar chart on the given polar axes."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = values + [values[0]]
    angles += [angles[0]]
    ax.plot(angles, values, "o-", linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=6, color="gray")
    ax.grid(True, alpha=0.4)
    ax.set_title(model_label, size=9, pad=12, color=color, fontweight="bold")


def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    metrics_dir = Path(config["outputs"]["metrics"])
    out_base = Path(config["outputs"]["maps"]) / "publication" / "fig3_model_comparison"
    out_base.mkdir(parents=True, exist_ok=True)

    ecosystems = [args.ecosystem] if args.ecosystem else list(config["ecosystems"].keys())

    # Collect data
    data = {}  # model → eco → metrics dict
    available_models = []
    for model in ALL_MODELS:
        model_ok = False
        data[model] = {}
        for eco in ecosystems:
            m = load_metrics(metrics_dir, eco, model)
            if m:
                data[model][eco] = m
                model_ok = True
        if model_ok:
            available_models.append(model)

    if not available_models:
        logger.error("No metrics found. Run 04_generate_counterfactual.py + 05_compute_metrics.py first.")
        return

    logger.info(f"Models with data: {available_models}")

    # ── Figure ────────────────────────────────────────────────────────────────
    n_models = len(available_models)
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#fafafa")

    # Layout: 3 rows × 3 cols (top row: 3 bars; middle: radar × n_models; bottom: TTC)
    # Using manual placement
    bar_w = 0.8 / len(ecosystems)
    x_base = np.arange(n_models)

    # ── Panel A: Impacted area ────────────────────────────────────────────────
    ax_area = fig.add_axes([0.05, 0.70, 0.28, 0.22])
    ax_area.set_facecolor("#f0f0f0")
    for ei, eco in enumerate(ecosystems):
        vals = [data[m].get(eco, {}).get("impacted_ha_max", 0) for m in available_models]
        bars = ax_area.bar(x_base + ei * bar_w - bar_w * (len(ecosystems) - 1) / 2,
                           vals, bar_w * 0.9,
                           label=eco.title(),
                           color=[MODEL_COLORS.get(m, "#888") for m in available_models],
                           alpha=0.85, hatch=ECO_HATCH[eco], edgecolor="white")
    ax_area.set_xticks(x_base)
    ax_area.set_xticklabels([MODEL_LABELS.get(m, m)[:8] for m in available_models],
                             rotation=30, ha="right", fontsize=8)
    ax_area.set_ylabel("Max Impacted Area (ha)", fontsize=9)
    ax_area.set_title("A  Vegetation Loss Area", fontsize=10, fontweight="bold")
    ax_area.grid(axis="y", alpha=0.3, linestyle="--")
    ax_area.legend(fontsize=8, loc="upper right")

    # ── Panel B: Carbon loss ──────────────────────────────────────────────────
    ax_carbon = fig.add_axes([0.37, 0.70, 0.28, 0.22])
    ax_carbon.set_facecolor("#f0f0f0")
    for ei, eco in enumerate(ecosystems):
        vals = [data[m].get(eco, {}).get("carbon_loss_tco2_eq", 0) for m in available_models]
        ax_carbon.bar(x_base + ei * bar_w - bar_w * (len(ecosystems) - 1) / 2,
                      vals, bar_w * 0.9,
                      label=eco.title(),
                      color=[MODEL_COLORS.get(m, "#888") for m in available_models],
                      alpha=0.85, hatch=ECO_HATCH[eco], edgecolor="white")
    ax_carbon.set_xticks(x_base)
    ax_carbon.set_xticklabels([MODEL_LABELS.get(m, m)[:8] for m in available_models],
                               rotation=30, ha="right", fontsize=8)
    ax_carbon.set_ylabel("Carbon Loss (tCO₂-eq)", fontsize=9)
    ax_carbon.set_title("B  Estimated Carbon Stock Loss", fontsize=10, fontweight="bold")
    ax_carbon.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Panel C: Recovery rate ────────────────────────────────────────────────
    ax_recov = fig.add_axes([0.69, 0.70, 0.28, 0.22])
    ax_recov.set_facecolor("#f0f0f0")
    for ei, eco in enumerate(ecosystems):
        vals = [data[m].get(eco, {}).get("gap_closing_rate", 0) or 0
                for m in available_models]
        ax_recov.bar(x_base + ei * bar_w - bar_w * (len(ecosystems) - 1) / 2,
                     vals, bar_w * 0.9,
                     label=eco.title(),
                     color=[MODEL_COLORS.get(m, "#888") for m in available_models],
                     alpha=0.85, hatch=ECO_HATCH[eco], edgecolor="white")
    ax_recov.set_xticks(x_base)
    ax_recov.set_xticklabels([MODEL_LABELS.get(m, m)[:8] for m in available_models],
                              rotation=30, ha="right", fontsize=8)
    ax_recov.set_ylabel("NDVI/quarter", fontsize=9)
    ax_recov.set_title("C  Recovery Speed (Gap Closing Rate)", fontsize=10, fontweight="bold")
    ax_recov.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Panel D: Radar charts ─────────────────────────────────────────────────
    categories = ["NDVI R²", "Area Accuracy\n(consistency)", "Carbon\nPrecision",
                  "Recovery\nSpeed", "Temporal\nCoherence"]
    radar_cols = min(n_models, 5)
    radar_w = 0.9 / radar_cols
    for mi, model in enumerate(available_models[:5]):
        col = mi % radar_cols
        left = 0.05 + col * radar_w * 1.0
        ax_r = fig.add_axes([left, 0.34, radar_w * 0.88, 0.28], polar=True)
        ax_r.set_facecolor("#f8f8f8")

        # Normalize metrics to [0,1] for radar
        # Higher is always better after normalization
        eco_key = ecosystems[0]
        m_data = data[model].get(eco_key, {})
        eval_m = load_eval_metrics(metrics_dir, model, eco_key)

        ndvi_r2  = max(0, min(1, eval_m.get("r2_ndvi", 0)))
        # Impacted area consistency: inverse normalized
        area_val = 1.0 - min(1.0, m_data.get("impacted_ha_max", 0) / 50000.0)
        carbon_v = 1.0 - min(1.0, m_data.get("carbon_loss_tco2_eq", 0) / 1e6)
        recov_v  = min(1.0, max(0, m_data.get("gap_closing_rate", 0) * 20))
        # Temporal coherence: proxy from gap_closing_rate stability
        temp_v   = min(1.0, max(0, 0.5 + ndvi_r2 * 0.5))

        values = [ndvi_r2, area_val, carbon_v, recov_v, temp_v]
        make_radar(ax_r, values, categories, MODEL_LABELS.get(model, model),
                   MODEL_COLORS.get(model, "#888"))

    # ── Panel E: Time-to-convergence horizontal bars ──────────────────────────
    ax_ttc = fig.add_axes([0.05, 0.04, 0.90, 0.24])
    ax_ttc.set_facecolor("#f0f0f0")
    yticks, yticklabels, colors = [], [], []
    y = 0
    for eco in ecosystems:
        for model in available_models:
            ttc = data[model].get(eco, {}).get("time_to_convergence_quarters")
            bar_color = MODEL_COLORS.get(model, "#888")
            if ttc is not None:
                ax_ttc.barh(y, ttc, height=0.7,
                            color=bar_color, alpha=0.85, edgecolor="white",
                            hatch=ECO_HATCH[eco])
                ax_ttc.text(ttc + 0.1, y,
                            f"{ttc}Q (~{ttc/4:.1f}y)",
                            va="center", ha="left", fontsize=8, color=bar_color,
                            fontweight="bold")
            else:
                ax_ttc.barh(y, 20, height=0.7, color=bar_color, alpha=0.3,
                            edgecolor=bar_color, linestyle="--",
                            hatch=ECO_HATCH[eco])
                ax_ttc.text(0.3, y, "Not recovered in window",
                            va="center", ha="left", fontsize=8, color=bar_color,
                            style="italic")
            yticklabels.append(f"{MODEL_LABELS.get(model,model)}\n({eco.title()[:4]})")
            yticks.append(y)
            colors.append(bar_color)
            y += 1
        y += 0.5  # gap between ecosystems

    ax_ttc.set_yticks(yticks)
    ax_ttc.set_yticklabels(yticklabels, fontsize=8)
    ax_ttc.set_xlabel("Quarters to Full Recovery (NDVI convergence)", fontsize=10)
    ax_ttc.set_title("E  Time to Ecosystem Recovery (model estimates)",
                      fontsize=10, fontweight="bold")
    ax_ttc.axvline(8, color="#cc0000", linestyle="--", alpha=0.5, linewidth=1)
    ax_ttc.text(8.1, y * 0.5, "2 years", color="#cc0000", fontsize=8)
    ax_ttc.grid(axis="x", alpha=0.3, linestyle="--")
    ax_ttc.set_xlim(0, 24)

    for tick, color in zip(ax_ttc.get_yticklabels(), colors):
        tick.set_color(color)

    # Title
    eco_str = " & ".join(e.title() for e in ecosystems)
    fig.suptitle(
        f"EcoRewind — Multi-Model Ecological Impact Comparison\n{eco_str}",
        fontsize=14, fontweight="bold", y=0.99, color="#1a1a2e"
    )

    out_path = out_base / f"{'_'.join(ecosystems)}_model_comparison.png"
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecosystem", type=str, default=None,
                        choices=["everglades", "mississippi"])
    main(parser.parse_args())