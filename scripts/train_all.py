"""
train_all.py
------------
Train all ECO-REWIND models back-to-back with one command.

Models trained in order: convlstm → unet → eco_transformer → unet_mamba → unet_patchtst
Dataloaders are built once and reused across all runs.

Usage:
    python scripts/train_all.py                                     # all 5 models
    python scripts/train_all.py --only unet_mamba unet_patchtst     # new models only
    python scripts/train_all.py --skip convlstm eco_transformer     # skip slow models
    python scripts/train_all.py --epochs 100                        # 100 epochs each
    python scripts/train_all.py --epochs-per-model 50,150,150,150,150
    python scripts/train_all.py --ecosystem everglades
    python scripts/train_all.py --no-wandb                          # disable W&B
"""

import sys
import os
import time
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry — all five models
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "convlstm": {
        "arch_key":   "convlstm",
        "import":     "models.convlstm",
        "class":      "ConvLSTMModel",
        "run_prefix": "convlstm",
        "label":      "ConvLSTM",
    },
    "unet": {
        "arch_key":   "unet_temporal",
        "import":     "models.unet_temporal",
        "class":      "UNetTemporalModel",
        "run_prefix": "unet_temporal",
        "label":      "UNet-Temporal",
    },
    "eco_transformer": {
        "arch_key":   "eco_transformer",
        "import":     "models.eco_transformer",
        "class":      "EcoTransformer",
        "run_prefix": "eco_transformer",
        "label":      "EcoTransformer",
    },
    "unet_mamba": {
        "arch_key":   "unet_mamba",
        "import":     "models.unet_mamba",
        "class":      "UNetMambaModel",
        "run_prefix": "unet_mamba",
        "label":      "UNet-Mamba",
    },
    "unet_patchtst": {
        "arch_key":   "unet_patchtst",
        "import":     "models.unet_patchtst",
        "class":      "UNetPatchTSTModel",
        "run_prefix": "unet_patchtst",
        "label":      "UNet-PatchTST",
    },
}

ALL_MODELS = ["convlstm", "unet", "eco_transformer", "unet_mamba", "unet_patchtst"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:   return f"{h}h {m}m {s}s"
    if m:   return f"{m}m {s}s"
    return f"{s}s"


def _build_model(name: str, config: Dict[str, Any]):
    info = MODEL_REGISTRY[name]
    import importlib
    mod = importlib.import_module(info["import"])
    cls = getattr(mod, info["class"])
    return cls(config)


def _ensure_model_config_defaults(name: str, config: Dict[str, Any]) -> None:
    """Insert default sub-config blocks for the two new models if absent."""
    if name == "unet_mamba" and "unet_mamba" not in config["model"]:
        config["model"]["unet_mamba"] = {
            "encoder_channels": [32, 64, 128, 256],
            "mamba_d_state": 16,
            "mamba_expand": 2,
            "n_mamba_layers": 2,
            "dropout": 0.1,
        }
        logger.info("  [unet_mamba] config block not in config.yaml — inserted defaults")

    if name == "unet_patchtst" and "unet_patchtst" not in config["model"]:
        config["model"]["unet_patchtst"] = {
            "encoder_channels": [32, 64, 128, 256],
            "patch_len": 2,
            "stride": 1,
            "n_transformer_layers": 2,
            "n_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
        }
        logger.info("  [unet_patchtst] config block not in config.yaml — inserted defaults")


def _free_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _print_separator(char: str = "=", width: int = 70) -> None:
    logger.info(char * width)


# ---------------------------------------------------------------------------
# Single model training run
# ---------------------------------------------------------------------------

def train_model(
    name: str,
    config: Dict[str, Any],
    train_loader,
    val_loader,
    ecosystem: str,
) -> Dict[str, Any]:
    info     = MODEL_REGISTRY[name]
    run_name = f"{info['run_prefix']}_{ecosystem}"

    _print_separator()
    logger.info(f"  Training: {info['label']}")
    logger.info(f"  Run name: {run_name}")
    logger.info(f"  Epochs:   {config['training']['epochs']}")
    _print_separator()

    result = {
        "model":    info["label"],
        "run_name": run_name,
        "status":   "failed",
        "best_val": float("inf"),
        "duration": 0.0,
        "metrics":  {},
        "error":    None,
    }

    t0 = time.time()
    try:
        config["model"]["architecture"] = info["arch_key"]
        _ensure_model_config_defaults(name, config)

        model = _build_model(name, config)
        n = model.count_parameters()
        logger.info(f"  Parameters: {n:,} ({n/1e6:.1f}M)")

        from models.losses import build_loss
        loss_fn = build_loss(config)

        from training.trainer import Trainer
        trainer = Trainer(
            model, loss_fn, train_loader, val_loader, config, run_name=run_name
        )
        trainer.train()
        result["best_val"]  = trainer.early_stop.best_loss
        result["best_ckpt"] = str(trainer.best_ckpt_path)

        from training.evaluator import ModelEvaluator
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        evaluator = ModelEvaluator(model, config, device=device)
        eval_metrics = evaluator.evaluate(val_loader)
        result["metrics"] = eval_metrics

        metrics_path = Path(config["outputs"]["metrics"]) / f"{run_name}_eval.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        result["metrics_path"] = str(metrics_path)
        result["status"] = "completed"

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  {info['label']} FAILED: {e}")
        logger.error(traceback.format_exc())

    result["duration"] = time.time() - t0
    _free_gpu()
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict[str, Any]]) -> None:
    _print_separator("=")
    logger.info("  TRAINING COMPLETE — Summary")
    _print_separator("=")

    col_w = [22, 12, 12, 12, 12, 12]
    header = (
        f"{'Model':<{col_w[0]}}"
        f"{'Status':<{col_w[1]}}"
        f"{'Best Val':<{col_w[2]}}"
        f"{'NDVI R²':<{col_w[3]}}"
        f"{'NDVI MSE':<{col_w[4]}}"
        f"{'Time':<{col_w[5]}}"
    )
    logger.info(f"  {header}")
    logger.info(f"  {'-' * sum(col_w)}")

    for r in results:
        ndvi_r2  = "—"
        ndvi_mse = "—"
        if r["metrics"]:
            ndvi_r2  = f"{r['metrics'].get('r2_ndvi',  float('nan')):.4f}"
            ndvi_mse = f"{r['metrics'].get('ndvi_mse', float('nan')):.6f}"

        best_val = f"{r['best_val']:.4f}" if r["best_val"] < 1e9 else "—"
        row = (
            f"{r['model']:<{col_w[0]}}"
            f"{r['status']:<{col_w[1]}}"
            f"{best_val:<{col_w[2]}}"
            f"{ndvi_r2:<{col_w[3]}}"
            f"{ndvi_mse:<{col_w[4]}}"
            f"{_fmt_time(r['duration']):<{col_w[5]}}"
        )
        logger.info(f"  {row}")

    _print_separator("=")

    completed = [r for r in results if r["status"] == "completed"]
    if completed:
        best = min(completed, key=lambda r: r["best_val"])
        logger.info(f"  Best model: {best['model']}  (val loss {best['best_val']:.4f})")
        logger.info(f"  Checkpoint: {best.get('best_ckpt', 'N/A')}")

    _print_separator("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # ---- Model selection ----
    if args.only:
        models_to_run = [m for m in ALL_MODELS if m in args.only]
    elif args.skip:
        models_to_run = [m for m in ALL_MODELS if m not in args.skip]
    else:
        models_to_run = list(ALL_MODELS)

    if not models_to_run:
        logger.error("No models selected. Check --only / --skip flags.")
        return

    # ---- Per-model epoch overrides ----
    epochs_per_model: Dict[str, int] = {}
    if args.epochs_per_model:
        parts = args.epochs_per_model.split(",")
        if len(parts) != len(ALL_MODELS):
            logger.error(
                f"--epochs-per-model expects {len(ALL_MODELS)} comma-separated values "
                f"({','.join(ALL_MODELS)}), got {len(parts)}"
            )
            return
        for model_name, ep in zip(ALL_MODELS, parts):
            epochs_per_model[model_name] = int(ep.strip())

    # ---- Global CLI overrides ----
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.no_wandb:
        config["wandb"]["enabled"] = False
        logger.info("  W&B logging disabled (--no-wandb)")

    ecosystem = args.ecosystem or config["training"].get("train_on", "joint")
    config["training"]["train_on"] = ecosystem

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _print_separator("=", 70)
    logger.info("  ECO-REWIND — Sequential Model Training")
    _print_separator("=", 70)
    logger.info(
        f"  Models:    {' → '.join(MODEL_REGISTRY[m]['label'] for m in models_to_run)}"
    )
    logger.info(f"  Ecosystem: {ecosystem}")
    logger.info(f"  Device:    {device}")
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  GPU:       {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB)")

    # ---- Build DataLoaders once ----
    logger.info("\n  Building dataloaders...")
    from datasets.eco_dataset import build_dataloaders

    processed_dir     = Path(config["data"]["processed_dir"])
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

    # ---- Train each model sequentially ----
    total_start = time.time()
    results     = []

    for i, model_name in enumerate(models_to_run):
        logger.info(
            f"\n\n{'='*70}\n"
            f"  [{i+1}/{len(models_to_run)}] Starting "
            f"{MODEL_REGISTRY[model_name]['label']}\n"
            f"{'='*70}"
        )

        # Deep-copy training config so per-model overrides don't bleed through
        import copy
        run_config = copy.deepcopy(config)

        if model_name in epochs_per_model:
            run_config["training"]["epochs"] = epochs_per_model[model_name]
            logger.info(f"  Epochs override: {epochs_per_model[model_name]}")

        result = train_model(
            model_name, run_config, train_loader, val_loader, ecosystem
        )
        results.append(result)

        if result["status"] == "completed":
            logger.info(
                f"\n  {MODEL_REGISTRY[model_name]['label']} done — "
                f"best val: {result['best_val']:.4f}, "
                f"time: {_fmt_time(result['duration'])}"
            )
        else:
            logger.warning(
                f"\n  {MODEL_REGISTRY[model_name]['label']} FAILED — "
                f"continuing to next model"
            )

    # ---- Summary ----
    _print_summary(results)

    summary_path = Path(config["outputs"]["metrics"]) / "train_all_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        safe = [
            {k: v for k, v in r.items() if k != "metrics"}
            for r in results
        ]
        json.dump(safe, f, indent=2)

    total_time = time.time() - total_start
    logger.info(f"\n  Total wall-clock time: {_fmt_time(total_time)}")
    logger.info(f"  Summary saved → {summary_path}")
    logger.info("\n  Next step: python scripts/04_generate_counterfactual.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all ECO-REWIND models sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (in fixed order):
  convlstm, unet, eco_transformer, unet_mamba, unet_patchtst

Examples:
  python scripts/train_all.py
  python scripts/train_all.py --only unet_mamba unet_patchtst
  python scripts/train_all.py --skip convlstm eco_transformer
  python scripts/train_all.py --epochs 100
  python scripts/train_all.py --epochs-per-model 50,150,150,150,150
  python scripts/train_all.py --ecosystem everglades --no-wandb
        """,
    )
    parser.add_argument(
        "--only", nargs="+",
        choices=ALL_MODELS,
        help="Train only these models (order preserved from ALL_MODELS)",
    )
    parser.add_argument(
        "--skip", nargs="+",
        choices=ALL_MODELS,
        help="Skip these models",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Epochs for all models (overrides config.yaml)",
    )
    parser.add_argument(
        "--epochs-per-model", type=str, default=None, dest="epochs_per_model",
        metavar="N,N,N,N,N",
        help=(
            "Per-model epoch counts in order: "
            "convlstm,unet,eco_transformer,unet_mamba,unet_patchtst"
        ),
    )
    parser.add_argument("--batch-size", type=int,   default=None, dest="batch_size")
    parser.add_argument("--ecosystem",  type=str,   default=None,
                        choices=["everglades", "mississippi", "joint"])
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--resume",     action="store_true",
                        help="Each trainer auto-resumes if a checkpoint exists")
    parser.add_argument("--no-wandb",   action="store_true", dest="no_wandb",
                        help="Disable W&B logging")
    main(parser.parse_args())