"""
viz_04_temporal_evolution.py
------------------------------
Figure 4: Temporal evolution grid — how the ecosystem changes quarter by quarter.

Shows a grid of NDVI frames:
  Row 0 (grey background): Pre-event context frames (last t_input quarters)
  Row 1 (green tint): Counterfactual prediction (no hurricane)
  Row 2 (red tint): Actual post-hurricane observations
  Row 3 (diverging): ΔNDVI damage per quarter

Each column = one quarter. Columns are labeled with the quarter date.
A red arrow marks the hurricane event. Color-coded recovery trajectory
overlaid as a subplot below the grid.

Usage:
    python scripts/viz_04_temporal_evolution.py
    python scripts/viz_04_temporal_evolution.py --model eco_transformer
    python scripts/viz_04_temporal_evolution.py --ecosystem everglades --n-quarters 8
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
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

ROW_META = [
    ("Pre-event Context",         "#e8e8e8", "Greys_r",    0, 1,   "Pre-event\n(actual)"),
    ("Counterfactual (No Storm)", "#d4edda",  "RdYlGn",   0, 0.7, "CF Pred.\n(no storm)"),
    ("Actual (Post-Hurricane)",   "#f8d7da",  "RdYlGn",   0, 0.7, "Actual\npost-event"),
    ("Damage (ΔNDVI)",            "#fff3cd",  "RdBu_r",  -0.3, 0.3, "Damage\n(CF − Act)"),
]


def quarter_label(t_event: int, start_year: int, start_q: int, offset: int) -> str:
    """Generate 'Q3 2017' style label from time index offset relative to t_event."""
    t = t_event + offset
    total_q = (start_year - 2016) * 4 + (start_q - 1) + t
    year = 2016 + total_q // 4
    q    = total_q % 4 + 1
    return f"Q{q}\n{year}"


def load_arrays(processed_dir: Path, eco: str, model: str):
    base = processed_dir / eco / model
    for f in ["counterfactual.npy", "actual_post.npy", "delta.npy"]:
        if not (base / f).exists():
            return None
    return {
        "cf":     np.load(str(base / "counterfactual.npy")),
        "actual": np.load(str(base / "actual_post.npy")),
        "delta":  np.load(str(base / "delta.npy")),
    }


def load_tensor_slice(processed_dir: Path, eco: str, t_input: int, t_event: int, ndvi_idx: int):
    """Load the pre-event context frames from the raw tensor."""
    tensor_path = processed_dir / eco / "tensor_normalized.npy"
    if not tensor_path.exists():
        tensor_path = processed_dir / eco / "tensor.npy"
    if not tensor_path.exists():
        return None
    tensor = np.load(str(tensor_path), mmap_mode="r")
    # Get last t_input frames before t_event
    start = max(0, t_event - t_input)
    pre = tensor[start:t_event, ndvi_idx]  # (t_input, H, W)
    return pre


def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    out_base = Path(config["outputs"]["maps"]) / "publication" / "fig4_temporal_evolution"
    out_base.mkdir(parents=True, exist_ok=True)

    ecosystems = [args.ecosystem] if args.ecosystem else list(config["ecosystems"].keys())
    models     = [args.model] if args.model else [
        m for m in ALL_MODELS
        if any((processed_dir / eco / m / "delta.npy").exists() for eco in ecosystems)
    ]

    ndvi_idx = config["bands"]["indices"]["ndvi"]
    t_input  = config["model"]["t_input"]

    for eco in ecosystems:
        eco_cfg = config["ecosystems"][eco]
        t_event = eco_cfg["t_event"]
        start_year = eco_cfg.get("start_year", 2016)
        start_q    = eco_cfg.get("start_quarter", 1)
        hurricane  = eco_cfg["hurricane"]
        hurricane_year = eco_cfg["hurricane_year"]

        # Load pre-event frames
        pre_frames = load_tensor_slice(processed_dir, eco, t_input, t_event, ndvi_idx)

        for model in models:
            arrays = load_arrays(processed_dir, eco, model)
            if arrays is None:
                logger.warning(f"  No arrays for {eco}/{model}")
                continue

            n_post = min(args.n_quarters, arrays["cf"].shape[0])
            n_pre  = pre_frames.shape[0] if pre_frames is not None else 0
            n_cols = n_pre + n_post

            logger.info(f"  {eco} / {model}: {n_pre} pre + {n_post} post quarters")

            # Build frame grids
            H, W = arrays["cf"].shape[2], arrays["cf"].shape[3]

            def get_frames(row_type):
                if row_type == "pre":
                    return [pre_frames[t] for t in range(n_pre)] if pre_frames is not None else []
                elif row_type == "cf":
                    return [arrays["cf"][t, ndvi_idx] for t in range(n_post)]
                elif row_type == "actual":
                    return [arrays["actual"][t, ndvi_idx] for t in range(n_post)]
                elif row_type == "delta":
                    return [arrays["delta"][t, ndvi_idx] for t in range(n_post)]

            # ── Figure ────────────────────────────────────────────────────
            n_rows = 4
            fig_w  = max(16, n_cols * 1.8)
            fig_h  = n_rows * 2.2 + 0.8
            fig = plt.figure(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor("#1a1a2e")

            gs = gridspec.GridSpec(
                n_rows, n_cols,
                figure=fig,
                hspace=0.05, wspace=0.04,
                left=0.07, right=0.98,
                top=0.90, bottom=0.08
            )

            row_labels = [
                "Pre-event\n(actual)",
                "Counterfactual\n(no hurricane)",
                "Actual\n(post-event)",
                "Damage\n(ΔNDVI)"
            ]
            row_cmaps  = ["Greys_r",  "RdYlGn", "RdYlGn", "RdBu_r"]
            row_vmins  = [0.0,        0.0,       0.0,       -0.3]
            row_vmaxs  = [1.0,        0.7,       0.7,        0.3]
            row_bg     = ["#2a2a3e", "#1a2e1a", "#2e1a1a", "#2e2a1a"]

            row_types = ["pre", "cf", "actual", "delta"]

            for ri, (rtype, rlabel, cmap, vmin, vmax, bg) in enumerate(zip(
                row_types, row_labels, row_cmaps, row_vmins, row_vmaxs, row_bg
            )):
                frames = get_frames(rtype)
                if not frames and rtype == "pre":
                    continue

                # Row label on left
                ax_label = fig.add_axes([0.01, 0.09 + (3 - ri) * 0.20, 0.055, 0.18])
                ax_label.set_facecolor(bg)
                ax_label.text(0.5, 0.5, rlabel,
                              transform=ax_label.transAxes,
                              ha="center", va="center", fontsize=8,
                              color="white", fontweight="bold",
                              rotation=0, multialignment="center")
                ax_label.axis("off")

                # Fill with post-quarter frames if "pre" has fewer than n_pre
                pre_offset = n_pre - len(frames) if rtype == "pre" else 0

                for ci in range(n_cols):
                    ax = fig.add_subplot(gs[ri, ci])
                    ax.set_facecolor(bg)

                    # For pre-event rows, only fill first n_pre columns
                    if rtype == "pre":
                        frame_ci = ci - pre_offset
                        if frame_ci < 0 or frame_ci >= len(frames):
                            ax.axis("off")
                            continue
                        frame = frames[frame_ci]
                    else:
                        # Post-event data fills columns n_pre onward
                        post_ci = ci - n_pre
                        if post_ci < 0 or post_ci >= len(frames):
                            ax.axis("off")
                            continue
                        frame = frames[post_ci]

                    ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax,
                              aspect="auto", interpolation="lanczos")
                    ax.axis("off")

                    # Column labels (top row only)
                    if ri == 0:
                        post_ci = ci - n_pre
                        if ci < n_pre:
                            lbl = f"t-{n_pre - ci}\n(pre)"
                            color = "#aaaaaa"
                        else:
                            lbl = f"+Q{post_ci+1}\n{quarter_label(t_event, start_year, start_q, post_ci)}"
                            color = "#ffcccc"
                        ax.set_title(lbl, fontsize=7, color=color, pad=3)

                    # Mark hurricane event column
                    if ci == n_pre - 1:
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#ff4444")
                            spine.set_linewidth(2)
                            spine.set_visible(True)

            # Hurricane marker
            event_x = (n_pre - 0.5) / n_cols
            fig.text(event_x, 0.93, f"↓ Hurricane {hurricane} ({hurricane_year})",
                     ha="center", va="bottom", fontsize=10, color="#ff4444",
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                               edgecolor="#ff4444", alpha=0.9))

            fig.suptitle(
                f"EcoRewind — {eco.title()} | {MODEL_LABELS.get(model, model)}\n"
                f"Temporal NDVI Evolution: Pre-event Context & Post-hurricane Predictions",
                color="white", fontsize=12, fontweight="bold", y=0.98
            )

            out_path = out_base / f"{eco}_{model}_temporal_grid.png"
            plt.savefig(str(out_path), dpi=160, bbox_inches="tight",
                        facecolor="#1a1a2e")
            plt.close()
            logger.info(f"  Saved: {out_path}")

    logger.info(f"\nFigures saved to: {out_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default=None)
    parser.add_argument("--ecosystem",  type=str, default=None,
                        choices=["everglades", "mississippi"])
    parser.add_argument("--n-quarters", type=int, default=8, dest="n_quarters",
                        help="Number of post-event quarters to show (default: 8)")
    main(parser.parse_args())