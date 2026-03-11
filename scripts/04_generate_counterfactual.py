"""
04_generate_counterfactual.py
------------------------------
Step 4: Load the best trained model and generate counterfactual
        ecological trajectory maps for both ecosystems.
 
Outputs:
  - outputs/maps/<ecosystem>/damage_maps_Q01.png
  - outputs/maps/<ecosystem>/delta_ndvi_impact_map.png
  - outputs/maps/<ecosystem>/trajectory_comparison.png
  - outputs/maps/<ecosystem>/recovery_metrics.png
  - processed/<ecosystem>/counterfactual.npy
  - processed/<ecosystem>/actual_post.npy
  - processed/<ecosystem>/delta.npy
 
Usage:
    python scripts/04_generate_counterfactual.py
    python scripts/04_generate_counterfactual.py --model convlstm
    python scripts/04_generate_counterfactual.py --model unet_temporal
    python scripts/04_generate_counterfactual.py --ecosystem everglades
    python scripts/04_generate_counterfactual.py --checkpoint outputs/checkpoints/.../best_model.pt
    python scripts/04_generate_counterfactual.py --n-forecast 12
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
    run_name = f"{model_arch}_{ecosystem}"
    run_dir = ckpt_base / run_name
    best = run_dir / "best_model.pt"
    if best.exists():
        return str(best)
    # Try joint training checkpoint
    joint_dir = ckpt_base / f"{model_arch}_joint"
    joint_best = joint_dir / "best_model.pt"
    if joint_best.exists():
        logger.info(f"  Using joint training checkpoint: {joint_best}")
        return str(joint_best)
    raise FileNotFoundError(
        f"No checkpoint found for {run_name}.\n"
        f"  Expected: {best}\n"
        f"  Run training first: python scripts/02_train_convlstm.py"
    )
 
 
def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
 
    model_arch = args.model or config["model"]["architecture"]
    config["model"]["architecture"] = model_arch
    n_forecast = args.n_forecast or 10   # default: 10 quarters post-event (~2.5 years)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("="*60)
    logger.info(" ECO-REWIND: Step 4 — Generate Counterfactual Maps")
    logger.info("="*60)
    logger.info(f"  Model:      {model_arch}")
    logger.info(f"  Forecast:   {n_forecast} quarters post-event")
    logger.info(f"  Device:     {device}")
 
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
    processed_dir = Path(config["data"]["processed_dir"])
    maps_dir = config["outputs"]["maps"]
    metrics_dir = config["outputs"]["metrics"]
 
    for eco in ecosystems:
        logger.info(f"\n{'='*50}")
        logger.info(f" Processing: {eco.upper()}")
        logger.info(f"{'='*50}")
 
        # --- Load tensor ---
        tensor_path = processed_dir / eco / "tensor.npy"
        validity_path = processed_dir / eco / "validity.npy"
        if not tensor_path.exists():
            logger.error(f"Tensor not found: {tensor_path}. Run 01_build_tensors.py first.")
            continue
 
        tensor = np.load(str(tensor_path))
        validity = np.load(str(validity_path))
        logger.info(f"  Tensor shape: {tensor.shape}")
 
        # --- Load normalizer ---
        norm_path = processed_dir / eco / "normalizer_stats.json"
        normalizer = EcoNormalizer(config)
        normalizer.load(str(norm_path))
 
        # --- Load model ---
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
 
        # --- Generate counterfactual ---
        generator = CounterfactualGenerator(model, config, normalizer, device=device)
        cf_results = generator.generate(eco, tensor, validity, n_forecast=n_forecast)
 
        # --- Save raw arrays ---
        eco_out_dir = processed_dir / eco
        for key in ["counterfactual", "actual_post", "delta"]:
            out_path = eco_out_dir / f"{key}.npy"
            np.save(str(out_path), cf_results[key])
            logger.info(f"  Saved {key} → {out_path}")
 
        # --- Compute metrics ---
        # Approximate pixel size from tensor dimensions
        eco_cfg = config["ecosystems"][eco.lower()]
        pixel_size_m = 10.0  # ~10m (EPSG:4326 at these latitudes, from QC reports)
 
        metrics_results = metrics_engine.compute_all(eco, cf_results, pixel_size_m)
 
        # Save metrics
        metrics_out = Path(metrics_dir) / f"{eco}_{model_arch}_metrics.json"
        metrics_engine.save(metrics_results, str(metrics_out))
 
        # --- Generate visualizations ---
        logger.info(f"\n  Generating visualizations...")
        plot_delta_maps(cf_results, config, eco, maps_dir, quarter_index=0)
        plot_trajectories(metrics_results, config, eco, maps_dir)
        plot_recovery_metrics(metrics_results, config, eco, maps_dir)
 
    logger.info("\n" + "="*60)
    logger.info(" Counterfactual generation complete!")
    logger.info(f"  Maps:    {maps_dir}/")
    logger.info(f"  Metrics: {metrics_dir}/")
    logger.info("="*60)
    logger.info("\nNext step: python scripts/05_compute_metrics.py")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactual maps")
    parser.add_argument("--model", type=str, default=None,
                        choices=["convlstm", "unet_temporal"],
                        help="Which model to use (default: from config)")
    parser.add_argument("--ecosystem", type=str, default=None,
                        choices=["everglades", "mississippi"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint (default: auto-find best)")
    parser.add_argument("--n-forecast", type=int, default=None,
                        help="Number of post-event quarters to forecast (default: 10)")
    main(parser.parse_args())