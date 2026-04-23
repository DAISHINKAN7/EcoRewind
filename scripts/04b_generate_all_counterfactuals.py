"""
04b_generate_all_counterfactuals.py
-------------------------------------
Runs 04_generate_counterfactual.py logic for ALL trained models
across ALL ecosystems in one shot.

For each model × ecosystem:
  → Finds best checkpoint
  → Generates counterfactual arrays
  → Computes metrics
  → Saves to processed/<eco>/<model>/

Then runs 05_compute_metrics.py --all-models to consolidate results.

Usage:
    python scripts/04b_generate_all_counterfactuals.py
    python scripts/04b_generate_all_counterfactuals.py --models eco_transformer unet_mamba
    python scripts/04b_generate_all_counterfactuals.py --ecosystem everglades
    python scripts/04b_generate_all_counterfactuals.py --n-forecast 10
    python scripts/04b_generate_all_counterfactuals.py --no-overwrite
"""

import sys
import os
import argparse
import logging
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import yaml
import numpy as np
import torch

ALL_MODELS = ["convlstm", "unet_temporal", "eco_transformer", "unet_mamba", "unet_patchtst"]


def find_best_checkpoint(config, model_arch: str, ecosystem: str) -> str:
    """Auto-find best checkpoint (joint or ecosystem-specific)."""
    ckpt_base = Path(config["outputs"]["checkpoints"])
    for run_name in [f"{model_arch}_{ecosystem}", f"{model_arch}_joint"]:
        best = ckpt_base / run_name / "best_model.pt"
        if best.exists():
            return str(best)
    raise FileNotFoundError(
        f"No checkpoint found for {model_arch} (tried {ecosystem} and joint).\n"
        f"  Run training first."
    )


def generate_for_model(config, model_arch: str, ecosystems: list,
                        n_forecast: int, no_overwrite: bool,
                        device: str) -> dict:
    """Run counterfactual generation for one model, all ecosystems."""
    from training.trainer import load_model
    from data.preprocessing.normalizer import EcoNormalizer
    from inference.counterfactual import CounterfactualGenerator
    from inference.metrics import EcologicalMetrics

    processed_dir = Path(config["data"]["processed_dir"])
    metrics_dir   = Path(config["outputs"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_engine = EcologicalMetrics(config)

    config_copy = dict(config)
    config_copy["model"] = dict(config["model"])
    config_copy["model"]["architecture"] = model_arch

    results = {}
    for eco in ecosystems:
        logger.info(f"\n  [{model_arch}] {eco.upper()}")

        eco_arrays_dir = processed_dir / eco / model_arch

        if no_overwrite and (eco_arrays_dir / "counterfactual.npy").exists():
            logger.info(f"    Skipping — arrays already exist (--no-overwrite)")
            results[eco] = "skipped"
            continue

        tensor_path   = processed_dir / eco / "tensor.npy"
        validity_path = processed_dir / eco / "validity.npy"
        if not tensor_path.exists():
            logger.error(f"    Tensor not found: {tensor_path}")
            results[eco] = "missing_tensor"
            continue

        try:
            ckpt_path = find_best_checkpoint(config, model_arch, eco)
        except FileNotFoundError as e:
            logger.error(f"    {e}")
            results[eco] = "missing_checkpoint"
            continue

        try:
            tensor   = np.load(str(tensor_path))
            validity = np.load(str(validity_path))

            norm_path  = processed_dir / eco / "normalizer_stats.json"
            normalizer = EcoNormalizer(config)
            normalizer.load(str(norm_path))

            logger.info(f"    Loading checkpoint: {ckpt_path}")
            model = load_model(config_copy, ckpt_path, device=device)

            generator = CounterfactualGenerator(model, config_copy, normalizer, device=device)
            cf_results = generator.generate(eco, tensor, validity, n_forecast=n_forecast)

            eco_arrays_dir.mkdir(parents=True, exist_ok=True)
            for key in ["counterfactual", "actual_post", "delta"]:
                np.save(str(eco_arrays_dir / f"{key}.npy"), cf_results[key])
            if "uncertainty" in cf_results:
                np.save(str(eco_arrays_dir / "uncertainty.npy"), cf_results["uncertainty"])

            logger.info(f"    Arrays saved to {eco_arrays_dir}/")

            metrics = metrics_engine.compute_all(eco, cf_results, pixel_size_m=10.0)
            metrics_engine.save(metrics, str(metrics_dir / f"{eco}_{model_arch}_metrics.json"))
            results[eco] = "success"

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"    FAILED: {e}")
            import traceback; traceback.print_exc()
            results[eco] = f"error: {e}"

    return results


def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 65)
    logger.info("  EcoRewind: Generate Counterfactuals — All Models")
    logger.info("=" * 65)
    logger.info(f"  Device: {device}")

    models     = args.models or ALL_MODELS
    ecosystems = [args.ecosystem] if args.ecosystem else list(config["ecosystems"].keys())
    n_forecast = args.n_forecast

    logger.info(f"  Models:      {models}")
    logger.info(f"  Ecosystems:  {ecosystems}")
    logger.info(f"  n_forecast:  {n_forecast}")
    logger.info(f"  no_overwrite: {args.no_overwrite}")

    all_results = {}
    for model in models:
        logger.info(f"\n{'='*50}\n  Model: {model}\n{'='*50}")
        r = generate_for_model(config, model, ecosystems, n_forecast,
                               args.no_overwrite, device)
        all_results[model] = r

    # Summary
    logger.info("\n" + "=" * 65)
    logger.info("  SUMMARY")
    logger.info("=" * 65)
    for model, eco_results in all_results.items():
        for eco, status in eco_results.items():
            status_icon = "✅" if status == "success" else ("⏭" if status == "skipped" else "❌")
            logger.info(f"  {status_icon} {model:20s} / {eco:15s} → {status}")

    logger.info("\n  Next step:")
    logger.info("    python scripts/05_compute_metrics.py --all-models")
    logger.info("    python scripts/viz_01_damage_comparison.py --all-models")
    logger.info("    python scripts/viz_02_recovery_trajectories.py --all-models")
    logger.info("    python scripts/viz_03_model_comparison.py")
    logger.info("    python scripts/viz_04_temporal_evolution.py --all-models")
    logger.info("    python scripts/viz_05_ecosystem_health_dashboard.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate counterfactuals for all models and ecosystems"
    )
    parser.add_argument(
        "--models", nargs="+", default=None, choices=ALL_MODELS,
        help="Which models to process (default: all trained)",
    )
    parser.add_argument(
        "--ecosystem", type=str, default=None,
        choices=["everglades", "mississippi"],
    )
    parser.add_argument(
        "--n-forecast", type=int, default=10, dest="n_forecast",
        help="Quarters to forecast past t_event (default: 10)",
    )
    parser.add_argument(
        "--no-overwrite", action="store_true", dest="no_overwrite",
        help="Skip if arrays already exist",
    )
    main(parser.parse_args())