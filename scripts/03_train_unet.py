"""
03_train_unet.py
----------------
Step 3: Train the U-Net + Temporal Attention model (primary architecture).
 
Run after 01_build_tensors.py (and optionally after 02_train_convlstm.py
for comparison). This is the main model used for counterfactual generation.
 
Usage:
    python scripts/03_train_unet.py
    python scripts/03_train_unet.py --epochs 100
    python scripts/03_train_unet.py --ecosystem everglades
    python scripts/03_train_unet.py --resume  # auto-resumes from latest checkpoint
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
        if total < 12e9:
            logger.warning(
                "  GPU has < 12 GB VRAM. Consider reducing batch_size to 2 "
                "or encoder_channels to [16, 32, 64, 128] in configs/config.yaml"
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
 
    config["model"]["architecture"] = "unet_temporal"
 
    logger.info("="*60)
    logger.info(" ECO-REWIND: Step 3 — Train U-Net + Temporal Attention")
    logger.info("="*60)
 
    enc_ch = config["model"]["unet_temporal"]["encoder_channels"]
    lstm_h = config["model"]["unet_temporal"]["lstm_hidden"]
    n_heads = config["model"]["unet_temporal"]["n_attention_heads"]
    logger.info(f"  Architecture:  U-Net (enc={enc_ch}) + LSTM({lstm_h}) + Attn({n_heads}h)")
    logger.info(f"  Train on:      {config['training']['train_on']}")
    logger.info(f"  Epochs:        {config['training']['epochs']}")
    logger.info(f"  Batch size:    {config['training']['batch_size']}")
    logger.info(f"  GPU:           {torch.cuda.is_available()}")
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
            logger.warning(f"  No patches for {eco}")
 
    if not ecosystem_patches:
        logger.error("No patches found. Run: python scripts/01_build_tensors.py")
        return
 
    train_loader, val_loader = build_dataloaders(config, ecosystem_patches)
 
    # --- Build model ---
    from models.unet_temporal import UNetTemporalModel
    from models.losses import build_loss
 
    model = UNetTemporalModel(config)
    n_params = model.count_parameters()
    logger.info(f"\n  Model parameters: {n_params:,} (~{n_params/1e6:.0f}M)")
 
    loss_fn = build_loss(config)
 
    # --- Train ---
    from training.trainer import Trainer
 
    run_name = f"unet_temporal_{config['training']['train_on']}"
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
    logger.info("\nNext step: python scripts/04_generate_counterfactual.py")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net + Temporal Attention")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ecosystem", type=str, default=None,
                        choices=["everglades", "mississippi", "joint"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint (automatic if found)")
    main(parser.parse_args())