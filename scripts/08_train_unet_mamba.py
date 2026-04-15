"""
08_train_unet_mamba.py
----------------------
Train UNet + Mamba (Selective State Space Model) — temporal module replacement.

Usage:
    python scripts/08_train_unet_mamba.py
    python scripts/08_train_unet_mamba.py --epochs 150
    python scripts/08_train_unet_mamba.py --ecosystem everglades
    python scripts/08_train_unet_mamba.py --resume
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_gpu_memory():
    if torch.cuda.is_available():
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"  GPU VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
        if total < 10e9:
            logger.warning(
                "  GPU has < 10 GB VRAM. If OOM, reduce encoder_channels to "
                "[16, 32, 64, 128] in configs/config.yaml under unet_mamba:"
            )


def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.ecosystem:
        config["training"]["train_on"] = args.ecosystem
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    config["model"]["architecture"] = "unet_mamba"

    # Ensure unet_mamba config block exists with sensible defaults
    if "unet_mamba" not in config["model"]:
        config["model"]["unet_mamba"] = {
            "encoder_channels": [32, 64, 128, 256],
            "mamba_d_state": 16,
            "mamba_expand": 2,
            "n_mamba_layers": 2,
            "dropout": 0.1,
        }
        logger.info("  unet_mamba config not found in config.yaml — using defaults")

    logger.info("=" * 60)
    logger.info(" ECO-REWIND: Step 8 — Train UNet + Mamba")
    logger.info("=" * 60)

    mb_cfg = config["model"]["unet_mamba"]
    logger.info(f"  Architecture:   UNet + Mamba SSM")
    logger.info(f"  encoder_ch:     {mb_cfg['encoder_channels']}")
    logger.info(f"  mamba_d_state:  {mb_cfg['mamba_d_state']}")
    logger.info(f"  mamba_expand:   {mb_cfg['mamba_expand']}")
    logger.info(f"  n_mamba_layers: {mb_cfg['n_mamba_layers']}")
    logger.info(f"  Train on:       {config['training']['train_on']}")
    logger.info(f"  Epochs:         {config['training']['epochs']}")
    logger.info(f"  Batch size:     {config['training']['batch_size']}")
    logger.info(f"  GPU available:  {torch.cuda.is_available()}")
    check_gpu_memory()

    # --- Build DataLoaders ---
    from datasets.eco_dataset import build_dataloaders

    processed_dir = Path(config["data"]["processed_dir"])
    ecosystem_patches = {}
    for eco in config["ecosystems"]:
        patch_dir = processed_dir / eco / "patches"
        if (patch_dir / "patches.npy").exists():
            ecosystem_patches[eco] = str(patch_dir)
        else:
            logger.warning(f"  No patches for {eco} — skipping")

    if not ecosystem_patches:
        logger.error("No patches found. Run: python scripts/01_build_tensors.py")
        return

    train_loader, val_loader = build_dataloaders(config, ecosystem_patches)
    logger.info(
        f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
    )

    # --- Build model ---
    from models.unet_mamba import UNetMambaModel
    from models.losses import build_loss

    model = UNetMambaModel(config)
    n_params = model.count_parameters()
    logger.info(f"\n  Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    loss_fn = build_loss(config)

    # --- Train ---
    from training.trainer import Trainer

    run_name = f"unet_mamba_{config['training']['train_on']}"
    trainer = Trainer(model, loss_fn, train_loader, val_loader, config, run_name=run_name)
    history = trainer.train()

    # --- Final evaluation ---
    logger.info("\n--- Final evaluation on validation set ---")
    from training.evaluator import ModelEvaluator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = ModelEvaluator(model, config, device=device)
    eval_metrics = evaluator.evaluate(val_loader)

    import json
    metrics_path = Path(config["outputs"]["metrics"]) / f"{run_name}_eval.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(f"\nMetrics saved → {metrics_path}")
    logger.info(f"Best checkpoint: {trainer.best_ckpt_path}")
    logger.info("\nNext step: python scripts/09_train_unet_patchtst.py")
    logger.info("  OR: python scripts/04_generate_counterfactual.py --model unet_mamba")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet + Mamba SSM")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None, dest="batch_size")
    parser.add_argument("--ecosystem",  type=str,   default=None,
                        choices=["everglades", "mississippi", "joint"])
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--resume",     action="store_true",
                        help="Auto-resume from latest checkpoint if found")
    main(parser.parse_args())