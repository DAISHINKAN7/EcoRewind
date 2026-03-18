"""
06_ablation.py
--------------
Step 6: Ablation study — train and evaluate all experiment variants.
 
Experiments (from your project doc):
  1. full         — Full model (S2 + SAR + L_eco) — baseline
  2. no_sar       — w/o SAR_VV band
  3. no_eco_loss  — w/o L_ecological
  4. no_temp_loss — w/o L_temporal_smooth
  5. ndvi_only    — NDVI-only (1 band)
  6. single_eco   — Train on Everglades only, infer on Mississippi
 
Each experiment trains from scratch and evaluates on val set.
Results compared in a bar chart.
 
Usage:
    python scripts/06_ablation.py                     # all experiments
    python scripts/06_ablation.py --experiment no_sar
    python scripts/06_ablation.py --model convlstm    # use ConvLSTM for ablation
    python scripts/06_ablation.py --epochs 30         # quick ablation
"""
 
import sys
import os
import argparse
import json
import logging
import copy
 
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
 
 
# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
 
EXPERIMENTS = {
    "full": {
        "description": "Full model: S2 + SAR + all losses",
        "modifications": {},
    },
    "no_sar": {
        "description": "w/o SAR_VV band (6 bands only)",
        "modifications": {"drop_sar": True},
    },
    "no_eco_loss": {
        "description": "w/o L_ecological constraint",
        "modifications": {"lambda_ecological": 0.0},
    },
    "no_temp_loss": {
        "description": "w/o L_temporal_smooth",
        "modifications": {"lambda_temporal": 0.0},
    },
    "ndvi_only": {
        "description": "NDVI-only input (1 band)",
        "modifications": {"ndvi_only": True},
    },
    "single_eco": {
        "description": "Train Everglades only → infer Mississippi",
        "modifications": {"train_on": "everglades"},
    },
}
 
 
def apply_modifications(config: dict, mods: dict) -> dict:
    """Apply experiment modifications to config."""
    cfg = copy.deepcopy(config)
 
    if mods.get("lambda_ecological") is not None:
        cfg["loss"]["lambda_ecological"] = mods["lambda_ecological"]
 
    if mods.get("lambda_temporal") is not None:
        cfg["loss"]["lambda_temporal"] = mods["lambda_temporal"]
 
    if mods.get("train_on"):
        cfg["training"]["train_on"] = mods["train_on"]
 
    if mods.get("drop_sar"):
        # Remove SAR_VV band from model input
        cfg["bands"]["count"] = 6
        cfg["bands"]["names"] = cfg["bands"]["names"][:6]
        # Adjust indices
        cfg["bands"]["indices"]["sar_vv"] = -1  # disabled
 
    if mods.get("ndvi_only"):
        # Only NDVI band (index 4)
        cfg["bands"]["count"] = 1
        cfg["bands"]["names"] = ["NDVI"]
        cfg["bands"]["indices"] = {
            "blue": -1, "green": -1, "red": -1, "nir": -1,
            "ndvi": 0, "ndwi": -1, "sar_vv": -1
        }
        cfg["bands"]["normalization"] = {"ndvi": "shift"}
 
    return cfg
 
 
def run_experiment(
    exp_name: str,
    exp_config: dict,
    base_config: dict,
    args,
) -> dict:
    """Train and evaluate one ablation experiment."""
    logger.info(f"\n{'='*60}")
    logger.info(f" ABLATION: {exp_name}")
    logger.info(f" {exp_config['description']}")
    logger.info(f"{'='*60}")
 
    mods = exp_config["modifications"]
    cfg = apply_modifications(base_config, mods)
    cfg["model"]["architecture"] = args.model
 
    # Shorter training for ablation
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
 
    # Unique run name for W&B
    cfg_copy = copy.deepcopy(cfg)
    if cfg_copy["wandb"]["enabled"]:
        import wandb
        # Finish any previous run
        try:
            wandb.finish()
        except Exception:
            pass
 
    # Build dataloaders with modified config
    from datasets.eco_dataset import build_dataloaders
 
    processed_dir = Path(cfg["data"]["processed_dir"])
 
    # Handle no_sar and ndvi_only: need to re-extract patches from modified tensor
    if mods.get("drop_sar") or mods.get("ndvi_only"):
        ecosystem_patches = _build_modified_patches(cfg, mods, processed_dir, exp_name)
    else:
        ecosystem_patches = {}
        for eco in cfg["ecosystems"]:
            patch_dir = processed_dir / eco / "patches"
            if (patch_dir / "patches.npy").exists():
                ecosystem_patches[eco] = str(patch_dir)
 
    if not ecosystem_patches:
        logger.warning(f"  [{exp_name}] No patches available — skipping")
        return {"best_val_loss": float("nan"), "ndvi_mse": float("nan")}
 
    train_loader, val_loader = build_dataloaders(cfg, ecosystem_patches)
 
    # Build model
    if args.model == "convlstm":
        from models.convlstm import ConvLSTMModel
        model = ConvLSTMModel(cfg)
    else:
        from models.unet_temporal import UNetTemporalModel
        model = UNetTemporalModel(cfg)
 
    from models.losses import build_loss
    from training.trainer import Trainer
    from training.evaluator import ModelEvaluator
 
    loss_fn = build_loss(cfg)
    run_name = f"ablation_{args.model}_{exp_name}"
 
    trainer = Trainer(model, loss_fn, train_loader, val_loader, cfg, run_name=run_name)
    trainer.train()
 
    # Evaluate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = ModelEvaluator(model, cfg, device=device)
    eval_metrics = evaluator.evaluate(val_loader)
 
    result = {
        "experiment": exp_name,
        "description": exp_config["description"],
        "best_val_loss": trainer.best_val_loss,
        "ndvi_mse": eval_metrics.get("ndvi_mse", float("nan")),
        "sar_mse": eval_metrics.get("sar_mse", float("nan")),
        "rmse_overall": eval_metrics.get("rmse_overall", float("nan")),
        "checkpoint": trainer.best_ckpt_path,
    }
 
    logger.info(
        f"\n  [{exp_name}] Results:\n"
        f"    best_val_loss: {result['best_val_loss']:.6f}\n"
        f"    ndvi_mse:      {result['ndvi_mse']:.6f}\n"
        f"    rmse_overall:  {result['rmse_overall']:.6f}"
    )
    return result
 
 
def _build_modified_patches(cfg, mods, processed_dir, exp_name):
    """
    For no_sar/ndvi_only experiments, extract patches from a band-sliced tensor.
    Saves to processed/<eco>/patches_<exp_name>/.
    """
    """
    IMPORTANT: sampler.build_patches() always writes to the shared
    processed/<eco>/patches/ directory. We backup that directory before
    building, copy results to the experiment-specific dir, then restore
    the backup so subsequent experiments see the original full-band patches.
    """
    import shutil
    from data.preprocessing.patch_sampler import PatchSampler
    from data.preprocessing.normalizer import EcoNormalizer
 
    sampler = PatchSampler(cfg)
    ecosystem_patches = {}
 
    for eco in cfg["ecosystems"]:
        tensor_path = processed_dir / eco / "tensor.npy"
        validity_path = processed_dir / eco / "validity.npy"
        norm_path = processed_dir / eco / "normalizer_stats.json"
 
        if not tensor_path.exists():
            continue
 
        tensor = np.load(str(tensor_path))
        validity = np.load(str(validity_path))
 
        # Apply band modifications
        if mods.get("drop_sar"):
            tensor = tensor[:, :6]      # Drop SAR_VV (last band)
            validity = validity
 
        if mods.get("ndvi_only"):
            ndvi_original_idx = 4
            tensor = tensor[:, ndvi_original_idx:ndvi_original_idx+1]
 
        # Re-normalize with modified band count
        norm = EcoNormalizer(cfg)
        t_event = cfg["ecosystems"][eco]["t_event"]
        norm.fit(tensor, t_event)
 
        norm_tensor = norm.transform(tensor)
 
        default_dir = processed_dir / eco / "patches"
        backup_dir  = processed_dir / eco / "patches_orig_backup"
        custom_dir  = processed_dir / eco / f"patches_{exp_name}"
 
        # --- Backup the shared patches/ dir before build_patches clobbers it ---
        if default_dir.exists():
            if backup_dir.exists():
                shutil.rmtree(str(backup_dir))
            shutil.copytree(str(default_dir), str(backup_dir))
 
        # Build modified patches (writes into default_dir / patches/)
        sampler.build_patches(
            ecosystem=eco,
            tensor=norm_tensor,
            validity=validity,
            normalizer=norm,
            force_rebuild=True,
        )
 
        # Copy freshly-built modified patches to experiment-specific dir
        custom_dir.mkdir(parents=True, exist_ok=True)
        if default_dir.exists():
            for f in default_dir.glob("*"):
                shutil.copy(str(f), str(custom_dir / f.name))
 
        # --- Restore the original patches so later experiments are unaffected ---
        if backup_dir.exists():
            if default_dir.exists():
                shutil.rmtree(str(default_dir))
            shutil.copytree(str(backup_dir), str(default_dir))
            shutil.rmtree(str(backup_dir))  # cleanup
 
        ecosystem_patches[eco] = str(custom_dir)
 
    return ecosystem_patches
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main(args):
    with open("configs/config.yaml") as f:
        base_config = yaml.safe_load(f)
 
    logger.info("="*60)
    logger.info(" ECO-REWIND: Step 6 — Ablation Study")
    logger.info("="*60)
 
    # Select experiments
    if args.experiment == "all":
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        if args.experiment not in EXPERIMENTS:
            logger.error(f"Unknown experiment: {args.experiment}")
            logger.error(f"Available: {list(EXPERIMENTS.keys())}")
            return
        experiments_to_run = [args.experiment]
 
    logger.info(f"  Experiments: {experiments_to_run}")
    logger.info(f"  Model:       {args.model}")
    logger.info(f"  Epochs:      {args.epochs or base_config['training']['epochs']}")
 
    # Run experiments
    all_results = {}
    for exp_name in experiments_to_run:
        try:
            result = run_experiment(
                exp_name,
                EXPERIMENTS[exp_name],
                base_config,
                args,
            )
            all_results[exp_name] = result
        except Exception as e:
            logger.error(f"  [{exp_name}] FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_results[exp_name] = {
                "experiment": exp_name,
                "error": str(e),
                "best_val_loss": float("nan"),
                "ndvi_mse": float("nan"),
            }
 
    # Save and visualize results
    metrics_dir = Path(base_config["outputs"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
 
    ablation_path = metrics_dir / f"ablation_{args.model}_results.json"
    with open(ablation_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: str(x))
    logger.info(f"\nAblation results → {ablation_path}")
 
    from visualization.maps import plot_ablation_comparison
    plot_ablation_comparison(all_results, str(base_config["outputs"]["maps"]))
 
    # Print summary table
    logger.info("\n" + "="*70)
    logger.info(" ABLATION SUMMARY")
    logger.info("="*70)
    logger.info(f"  {'Experiment':>20}  {'Val Loss':>10}  {'NDVI MSE':>10}  Description")
    logger.info("  " + "-"*65)
    for name, r in all_results.items():
        vl = r.get("best_val_loss", float("nan"))
        nm = r.get("ndvi_mse", float("nan"))
        desc = EXPERIMENTS[name]["description"][:35]
        logger.info(f"  {name:>20}  {vl:>10.6f}  {nm:>10.6f}  {desc}")
    logger.info("="*70)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Which ablation experiment to run"
    )
    parser.add_argument(
        "--model", type=str, default="convlstm",
        choices=["convlstm", "unet_temporal"],
        help="Model architecture to use for ablation"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Epochs per ablation experiment (default: 30 for speed)"
    )
    main(parser.parse_args())