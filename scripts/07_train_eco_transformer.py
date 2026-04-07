"""
07_train_eco_transformer.py
---------------------------
Step 7: Train EcoTransformer — factorized spatiotemporal attention model.
 
EcoTransformer uses:
  - CNN spatial encoder (128px → 16px token grid)
  - Factorized temporal + spatial self-attention
  - Cross-attention prediction head for T_out future steps
 
Expected to outperform ConvLSTM and UNet-Temporal via global spatial context
and flexible temporal dependency modelling without recurrence.
 
Usage:
    python scripts/07_train_eco_transformer.py
    python scripts/07_train_eco_transformer.py --epochs 150
    python scripts/07_train_eco_transformer.py --ecosystem everglades
    python scripts/07_train_eco_transformer.py --embed-dim 192 --depth 6
    python scripts/07_train_eco_transformer.py --resume
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
        if total < 16e9:
            logger.warning(
                "  GPU has < 16 GB VRAM. If OOM, try:\n"
                "    --embed-dim 96  (halves token dimension)\n"
                "    --depth 2       (fewer transformer layers)\n"
                "    or reduce batch_size in configs/config.yaml"
            )
 
 
def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
 
    # CLI overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.ecosystem:
        config["training"]["train_on"] = args.ecosystem
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.embed_dim:
        config["model"]["eco_transformer"]["embed_dim"] = args.embed_dim
    if args.depth:
        config["model"]["eco_transformer"]["depth"] = args.depth
    if args.n_heads:
        config["model"]["eco_transformer"]["n_heads"] = args.n_heads
 
    config["model"]["architecture"] = "eco_transformer"
 
    logger.info("=" * 60)
    logger.info(" ECO-REWIND: Step 7 — Train EcoTransformer")
    logger.info("=" * 60)
 
    et_cfg = config["model"]["eco_transformer"]
    logger.info(f"  Architecture:   EcoTransformer")
    logger.info(f"  embed_dim:      {et_cfg['embed_dim']}")
    logger.info(f"  depth:          {et_cfg['depth']}")
    logger.info(f"  n_heads:        {et_cfg['n_heads']}")
    logger.info(f"  mlp_ratio:      {et_cfg['mlp_ratio']}")
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
    from models.eco_transformer import build_eco_transformer
    from models.losses import build_loss
 
    model = build_eco_transformer(config)
    loss_fn = build_loss(config)
 
    # --- Train ---
    from training.trainer import Trainer
 
    run_name = f"eco_transformer_{config['training']['train_on']}"
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
    parser = argparse.ArgumentParser(description="Train EcoTransformer")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None, dest="batch_size")
    parser.add_argument("--ecosystem",  type=str,   default=None,
                        choices=["everglades", "mississippi", "joint"])
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--embed-dim",  type=int,   default=None, dest="embed_dim",
                        help="Token embedding dimension (default: 128)")
    parser.add_argument("--depth",      type=int,   default=None,
                        help="Number of transformer encoder layers (default: 4)")
    parser.add_argument("--n-heads",    type=int,   default=None, dest="n_heads",
                        help="Attention heads — embed_dim must be divisible (default: 8)")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from latest checkpoint (automatic if found)")
    main(parser.parse_args())