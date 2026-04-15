"""
09_train_unet_patchtst.py
--------------------------
Train UNet + PatchTST-style Temporal Transformer.

Usage:
    python scripts/09_train_unet_patchtst.py
    python scripts/09_train_unet_patchtst.py --epochs 150
    python scripts/09_train_unet_patchtst.py --ecosystem everglades
    python scripts/09_train_unet_patchtst.py --patch-len 2 --n-layers 3
    python scripts/09_train_unet_patchtst.py --resume
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
                "[16, 32, 64, 128] in configs/config.yaml under unet_patchtst:"
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

    # CLI overrides for PatchTST-specific hyperparams
    if "unet_patchtst" not in config["model"]:
        config["model"]["unet_patchtst"] = {
            "encoder_channels": [32, 64, 128, 256],
            "patch_len": 2,
            "stride": 1,
            "n_transformer_layers": 2,
            "n_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
        }
        logger.info("  unet_patchtst config not found in config.yaml — using defaults")

    if args.patch_len is not None:
        config["model"]["unet_patchtst"]["patch_len"] = args.patch_len
    if args.n_layers is not None:
        config["model"]["unet_patchtst"]["n_transformer_layers"] = args.n_layers
    if args.n_heads is not None:
        config["model"]["unet_patchtst"]["n_heads"] = args.n_heads

    config["model"]["architecture"] = "unet_patchtst"

    logger.info("=" * 60)
    logger.info(" ECO-REWIND: Step 9 — Train UNet + PatchTST Transformer")
    logger.info("=" * 60)

    pt_cfg = config["model"]["unet_patchtst"]
    logger.info(f"  Architecture:        UNet + PatchTST")
    logger.info(f"  encoder_ch:          {pt_cfg['encoder_channels']}")
    logger.info(f"  patch_len:           {pt_cfg['patch_len']}")
    logger.info(f"  stride:              {pt_cfg['stride']}")
    logger.info(f"  n_transformer_layers:{pt_cfg['n_transformer_layers']}")
    logger.info(f"  n_heads:             {pt_cfg['n_heads']}")
    logger.info(f"  mlp_ratio:           {pt_cfg['mlp_ratio']}")
    logger.info(f"  Train on:            {config['training']['train_on']}")
    logger.info(f"  Epochs:              {config['training']['epochs']}")
    logger.info(f"  Batch size:          {config['training']['batch_size']}")
    logger.info(f"  GPU available:       {torch.cuda.is_available()}")
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
    from models.unet_patchtst import UNetPatchTSTModel
    from models.losses import build_loss

    model = UNetPatchTSTModel(config)
    n_params = model.count_parameters()
    logger.info(f"\n  Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    loss_fn = build_loss(config)

    # --- Train ---
    from training.trainer import Trainer

    run_name = f"unet_patchtst_{config['training']['train_on']}"
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
    logger.info("\nNext step: python scripts/04_generate_counterfactual.py --model unet_patchtst")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet + PatchTST Temporal Transformer")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None, dest="batch_size")
    parser.add_argument("--ecosystem",  type=str,   default=None,
                        choices=["everglades", "mississippi", "joint"])
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--patch-len",  type=int,   default=None, dest="patch_len",
                        help="Temporal patch length K (default: 2)")
    parser.add_argument("--n-layers",   type=int,   default=None, dest="n_layers",
                        help="Number of transformer encoder layers (default: 2)")
    parser.add_argument("--n-heads",    type=int,   default=None, dest="n_heads",
                        help="Attention heads (default: 4)")
    parser.add_argument("--resume",     action="store_true",
                        help="Auto-resume from latest checkpoint if found")
    main(parser.parse_args())