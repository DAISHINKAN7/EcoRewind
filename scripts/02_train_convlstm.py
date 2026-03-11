"""
02_train_convlstm.py
--------------------
Step 2: Train the ConvLSTM baseline model.
 
Run after 01_build_tensors.py.
 
Usage:
    python scripts/02_train_convlstm.py
    python scripts/02_train_convlstm.py --epochs 50
    python scripts/02_train_convlstm.py --ecosystem everglades
    python scripts/02_train_convlstm.py --batch-size 8
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
 
 
def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    config["training"].setdefault("train_on", "joint")
 
    # Override config with CLI args
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.ecosystem:
        config["training"]["train_on"] = args.ecosystem
    if args.lr:
        config["training"]["learning_rate"] = args.lr
 
    config["model"]["architecture"] = "convlstm"
 
    logger.info("="*60)
    logger.info(" ECO-REWIND: Step 2 — Train ConvLSTM Baseline")
    logger.info("="*60)
    logger.info(f"  Architecture:  ConvLSTM ({config['model']['convlstm']['hidden_channels']})")
    logger.info(f"  Train on:      {config['training']['train_on']}")
    logger.info(f"  Epochs:        {config['training']['epochs']}")
    logger.info(f"  Batch size:    {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  GPU:           {torch.cuda.is_available()}")
 
    # --- Build DataLoaders ---
    from datasets.eco_dataset import build_dataloaders
 
    processed_dir = Path(config["data"]["processed_dir"])
    ecosystem_patches = {}
    for eco in config["ecosystems"]:
        patch_dir = processed_dir / eco / "patches"
        if (patch_dir / "patches.npy").exists():
            ecosystem_patches[eco] = str(patch_dir)
        else:
            logger.warning(f"  No patches for {eco} — run 01_build_tensors.py first")
 
    if not ecosystem_patches:
        logger.error("No patches found. Run: python scripts/01_build_tensors.py")
        return
 
    train_loader, val_loader = build_dataloaders(config, ecosystem_patches)
 
    # --- Build model ---
    from models.convlstm import ConvLSTMModel
    from models.losses import build_loss
 
    model = ConvLSTMModel(config)
    n_params = model.count_parameters()
    logger.info(f"\n  Model parameters: {n_params:,}")
 
    loss_fn = build_loss(config)
 
    # --- Train ---
    from training.trainer import Trainer
 
    active_ecos = [
        eco for eco, enabled in
        config["training"].get("ecosystems_for_training", {}).items()
        if enabled
    ] or list(ecosystem_patches.keys())
    run_name = f"convlstm_{'_'.join(active_ecos)}"
    trainer = Trainer(model, loss_fn, train_loader, val_loader, config, run_name=run_name)
 
    history = trainer.train()
 
    # --- Final evaluation ---
    logger.info("\n--- Final evaluation on validation set ---")
    from training.evaluator import ModelEvaluator
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = ModelEvaluator(model, config, device=device)
    eval_metrics = evaluator.evaluate(val_loader)
 
    # Save eval metrics
    import json
    metrics_path = Path(config["outputs"]["metrics"]) / f"{run_name}_eval.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
 
    logger.info(f"\nMetrics saved → {metrics_path}")
    logger.info(f"Best checkpoint: {trainer.best_ckpt_path}")
    logger.info("\nNext step: python scripts/03_train_unet.py")
    logger.info("  OR skip to: python scripts/04_generate_counterfactual.py")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvLSTM baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ecosystem", type=str, default=None,
                        choices=["everglades", "mississippi", "joint"])
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    main(parser.parse_args())