"""
04_generate_counterfactual.py
------------------------------
Step 4: Load the best trained model and generate counterfactual
        ecological trajectory maps for both ecosystems.

BUG FIX: All output paths are now model-specific.
Previously: processed/<eco>/counterfactual.npy   (overwritten by every run)
Now:        processed/<eco>/<model_arch>/counterfactual.npy

This means you can safely run this for unet_patchtst and unet_mamba
without losing either set of results.

05_compute_metrics.py is updated to accept a --model flag and reads
from the model-specific subdirectory.

Outputs (per model):
  - processed/<eco>/<model_arch>/counterfactual.npy
  - processed/<eco>/<model_arch>/actual_post.npy
  - processed/<eco>/<model_arch>/delta.npy
  - outputs/maps/<eco>/<model_arch>/damage_maps_Q01.png
  - outputs/maps/<eco>/<model_arch>/trajectory_comparison.png
  - outputs/metrics/<eco>_<model_arch>_metrics.json

Usage:
    python scripts/04_generate_counterfactual.py
    python scripts/04_generate_counterfactual.py --model unet_patchtst
    python scripts/04_generate_counterfactual.py --model unet_mamba
    python scripts/04_generate_counterfactual.py --ecosystem everglades
    python scripts/04_generate_counterfactual.py --checkpoint outputs/checkpoints/.../best_model.pt
    python scripts/04_generate_counterfactual.py --n-forecast 12
    python scripts/04_generate_counterfactual.py --no-overwrite  # skip if arrays exist
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(config, model_arch: str, ecosystem: str) -> str:
    """Auto-find the best checkpoint for a given model+ecosystem combo."""
    ckpt_base = Path(config["outputs"]["checkpoints"])

    # Try ecosystem-specific first
    run_name = f"{model_arch}_{ecosystem}"
    run_dir  = ckpt_base / run_name
    best     = run_dir / "best_model.pt"
    if best.exists():
        return str(best)

    # Fall back to joint training checkpoint
    joint_dir  = ckpt_base / f"{model_arch}_joint"
    joint_best = joint_dir / "best_model.pt"
    if joint_best.exists():
        logger.info(f"  Using joint training checkpoint: {joint_best}")
        return str(joint_best)

    raise FileNotFoundError(
        f"No checkpoint found for {run_name}.\n"
        f"  Tried: {best}\n"
        f"       : {joint_best}\n"
        f"  Run training first."
    )


def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model_arch = args.model or config["model"]["architecture"]
    config["model"]["architecture"] = model_arch
    n_forecast = args.n_forecast or 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 60)
    logger.info(" ECO-REWIND: Step 4 — Generate Counterfactual Maps")
    logger.info("=" * 60)
    logger.info(f"  Model:       {model_arch}")
    logger.info(f"  Forecast:    {n_forecast} quarters post-event")
    logger.info(f"  Device:      {device}")
    logger.info(f"  No-overwrite: {args.no_overwrite}")

    ecosystems = (
        [args.ecosystem] if args.ecosystem
        else list(config["ecosystems"].keys())
    )

    from training.trainer import load_model
    from data.preprocessing.normalizer import EcoNormalizer
    from inference.counterfactual import CounterfactualGenerator
    from inference.metrics import EcologicalMetrics
    from visualization.maps import (
        plot_delta_maps, plot_trajectories, plot_recovery_metrics
    )

    metrics_engine = EcologicalMetrics(config)
    processed_dir  = Path(config["data"]["processed_dir"])
    maps_dir_base  = Path(config["outputs"]["maps"])
    metrics_dir    = Path(config["outputs"]["metrics"])

    for eco in ecosystems:
        logger.info(f"\n{'='*50}")
        logger.info(f" Processing: {eco.upper()} with {model_arch}")
        logger.info(f"{'='*50}")

        # ── Model-specific output directories ─────────────────────────────
        eco_arrays_dir = processed_dir / eco / model_arch
        eco_maps_dir   = maps_dir_base / eco / model_arch

        # Check --no-overwrite
        if args.no_overwrite:
            cf_path = eco_arrays_dir / "counterfactual.npy"
            if cf_path.exists():
                logger.info(
                    f"  Skipping {eco}/{model_arch} — arrays already exist.\n"
                    f"  (use without --no-overwrite to regenerate)"
                )
                continue

        # ── Load tensor ───────────────────────────────────────────────────
        tensor_path   = processed_dir / eco / "tensor.npy"
        validity_path = processed_dir / eco / "validity.npy"
        if not tensor_path.exists():
            logger.error(f"Tensor not found: {tensor_path}. Run 01_build_tensors.py first.")
            continue

        tensor   = np.load(str(tensor_path))
        validity = np.load(str(validity_path))
        logger.info(f"  Tensor shape: {tensor.shape}")

        # ── Load normalizer ───────────────────────────────────────────────
        norm_path  = processed_dir / eco / "normalizer_stats.json"
        normalizer = EcoNormalizer(config)
        normalizer.load(str(norm_path))

        # ── Load model ────────────────────────────────────────────────────
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            try:
                ckpt_path = find_best_checkpoint(config, model_arch, eco)
            except FileNotFoundError as e:
                logger.error(str(e))
                continue

        logger.info(f"  Loading model from: {ckpt_path}")
        model = load_model(config, ckpt_path, device=device)

        # ── Generate counterfactual ───────────────────────────────────────
        generator  = CounterfactualGenerator(model, config, normalizer, device=device)
        cf_results = generator.generate(eco, tensor, validity, n_forecast=n_forecast)

        # ── Save raw arrays (model-specific path) ─────────────────────────
        eco_arrays_dir.mkdir(parents=True, exist_ok=True)
        for key in ["counterfactual", "actual_post", "delta"]:
            out_path = eco_arrays_dir / f"{key}.npy"
            np.save(str(out_path), cf_results[key])
            logger.info(f"  Saved {key} → {out_path}")

        if "uncertainty" in cf_results:
            unc_path = eco_arrays_dir / "uncertainty.npy"
            np.save(str(unc_path), cf_results["uncertainty"])
            logger.info(f"  Saved uncertainty → {unc_path}")

        # ── Compute metrics ───────────────────────────────────────────────
        pixel_size_m  = 10.0
        metrics_results = metrics_engine.compute_all(eco, cf_results, pixel_size_m)

        metrics_out = metrics_dir / f"{eco}_{model_arch}_metrics.json"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_engine.save(metrics_results, str(metrics_out))
        logger.info(f"  Metrics saved → {metrics_out}")

        # ── Generate visualizations (model-specific map dir) ──────────────
        logger.info(f"\n  Generating visualizations in {eco_maps_dir}...")
        eco_maps_dir.mkdir(parents=True, exist_ok=True)
        plot_delta_maps(cf_results, config, eco, str(maps_dir_base / eco), quarter_index=0)
        plot_trajectories(metrics_results, config, eco, str(maps_dir_base / eco))
        plot_recovery_metrics(metrics_results, config, eco, str(maps_dir_base / eco))

    logger.info("\n" + "=" * 60)
    logger.info(" Counterfactual generation complete!")
    logger.info(f"  Arrays: processed/<eco>/<model>/")
    logger.info(f"  Maps:   outputs/maps/<eco>/<model>/")
    logger.info(f"  Metrics: {metrics_dir}/")
    logger.info("=" * 60)
    logger.info("\nNext step: python scripts/05_compute_metrics.py --model " + model_arch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactual maps")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["convlstm", "unet_temporal", "eco_transformer", "unet_mamba", "unet_patchtst"],
        help="Which model to use (default: from config.yaml)",
    )
    parser.add_argument(
        "--ecosystem", type=str, default=None,
        choices=["everglades", "mississippi"],
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a specific checkpoint (default: auto-find best)",
    )
    parser.add_argument(
        "--n-forecast", type=int, default=None,
        help="Quarters to forecast past t_event (default: 10)",
    )
    parser.add_argument(
        "--no-overwrite", action="store_true", dest="no_overwrite",
        help="Skip generation if model-specific arrays already exist",
    )
    main(parser.parse_args())