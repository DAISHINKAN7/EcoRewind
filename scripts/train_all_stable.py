"""
train_all_stable.py
-------------------
Replacement for scripts/train_all.py.

Wires together all stability fixes:
  1. Applies config_patch (reduced LR, SSIM warmup, etc.)
  2. Reduces nan_threshold (eliminates NaN batch skips)
  3. Patches EcoTimeSeriesDataset.__getitem__ with band-aware fills
  4. Uses the stable Trainer from trainer.py (fixed scheduler order)

Usage:
    python scripts/train_all_stable.py --only unet_mamba unet_patchtst --no-wandb
    python scripts/train_all_stable.py                      # all 5 models
    python scripts/train_all_stable.py --skip convlstm unet eco_transformer
"""

import sys
import os
import time
import json
import copy
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List

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
# Import stability patches
# ---------------------------------------------------------------------------

# These imports assume eco_rewind_fixes/ is in sys.path or copied to project root
try:
    from config_patch import apply_stability_patch
    from eco_dataset_patch import make_patched_getitem, diagnose_patches
except ImportError:
    # Fallback: patches not available, warn and use identity
    logger.warning(
        "Stability patches not found. Copy eco_rewind_fixes/ to project root."
    )
    def apply_stability_patch(c): return c
    def make_patched_getitem(c): return None
    def diagnose_patches(*a, **k): return {}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "convlstm":        {"arch_key": "convlstm",        "import": "models.convlstm",      "class": "ConvLSTMModel",      "label": "ConvLSTM"},
    "unet":            {"arch_key": "unet_temporal",    "import": "models.unet_temporal", "class": "UNetTemporalModel",  "label": "UNet-Temporal"},
    "eco_transformer": {"arch_key": "eco_transformer",  "import": "models.eco_transformer","class": "EcoTransformer",    "label": "EcoTransformer"},
    "unet_mamba":      {"arch_key": "unet_mamba",       "import": "models.unet_mamba",    "class": "UNetMambaModel",     "label": "UNet-Mamba"},
    "unet_patchtst":   {"arch_key": "unet_patchtst",    "import": "models.unet_patchtst", "class": "UNetPatchTSTModel",  "label": "UNet-PatchTST"},
}

ALL_MODELS = ["convlstm", "unet", "eco_transformer", "unet_mamba", "unet_patchtst"]


def _fmt_time(s: float) -> str:
    h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")


def _build_model(name: str, config: Dict):
    import importlib
    info = MODEL_REGISTRY[name]
    mod  = importlib.import_module(info["import"])
    cls  = getattr(mod, info["class"])
    return cls(config)


def _ensure_model_defaults(name: str, config: Dict):
    if name == "unet_mamba" and "unet_mamba" not in config["model"]:
        config["model"]["unet_mamba"] = {
            "encoder_channels": [32, 64, 128, 256],
            "mamba_d_state": 16, "mamba_expand": 2,
            "n_mamba_layers": 2, "dropout": 0.1,
        }
    if name == "unet_patchtst" and "unet_patchtst" not in config["model"]:
        config["model"]["unet_patchtst"] = {
            "encoder_channels": [32, 64, 128, 256],
            "patch_len": 2, "stride": 1,
            "n_transformer_layers": 3, "n_heads": 4,
            "mlp_ratio": 2.0, "dropout": 0.1,
        }


def _free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_model(
    name: str, config: Dict, train_loader, val_loader, ecosystem: str
) -> Dict:
    info     = MODEL_REGISTRY[name]
    run_name = f"{info['arch_key']}_{ecosystem}"

    logger.info(f"\n{'='*70}\n  Training: {info['label']}  ({run_name})\n{'='*70}")

    result = {
        "model": info["label"], "run_name": run_name,
        "status": "failed", "best_val": float("inf"),
        "duration": 0.0, "metrics": {}, "error": None,
    }
    t0 = time.time()

    try:
        config["model"]["architecture"] = info["arch_key"]
        _ensure_model_defaults(name, config)

        model = _build_model(name, config)
        n = model.count_parameters()
        logger.info(f"  Parameters: {n:,} ({n/1e6:.1f}M)")

        from models.losses import build_loss
        loss_fn = build_loss(config)

        from training.trainer import Trainer
        trainer = Trainer(model, loss_fn, train_loader, val_loader, config, run_name=run_name)
        trainer.train()

        result["best_val"]  = trainer.early_stop.best_loss
        result["best_ckpt"] = str(trainer.best_ckpt_path)

        from training.evaluator import ModelEvaluator
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        evaluator = ModelEvaluator(model, config, device=device)
        eval_metrics = evaluator.evaluate(val_loader)
        result["metrics"] = eval_metrics

        mp = Path(config["outputs"]["metrics"]) / f"{run_name}_eval.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        with open(mp, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        result["status"] = "completed"

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  {info['label']} FAILED: {e}\n{traceback.format_exc()}")

    result["duration"] = time.time() - t0
    _free_gpu()
    return result


def _print_summary(results: List[Dict]):
    logger.info(f"\n{'='*70}\n  TRAINING COMPLETE — Summary\n{'='*70}")
    header = f"{'Model':<22} {'Status':<12} {'Best Val':<12} {'NDVI R²':<10} {'SAR R²':<10} {'Time'}"
    logger.info(f"  {header}")
    logger.info(f"  {'-'*78}")
    for r in results:
        ndvi_r2 = f"{r['metrics'].get('r2_ndvi',  float('nan')):.4f}" if r["metrics"] else "—"
        sar_r2  = f"{r['metrics'].get('r2_sar_vv', float('nan')):.4f}" if r["metrics"] else "—"
        bv = f"{r['best_val']:.4f}" if r["best_val"] < 1e9 else "—"
        logger.info(f"  {r['model']:<22} {r['status']:<12} {bv:<12} {ndvi_r2:<10} {sar_r2:<10} {_fmt_time(r['duration'])}")
    logger.info("="*70)
    completed = [r for r in results if r["status"] == "completed"]
    if completed:
        best = min(completed, key=lambda r: r["best_val"])
        logger.info(f"  Best model: {best['model']}  (val {best['best_val']:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # ── Apply stability patch ────────────────────────────────────────────────
    config = apply_stability_patch(config)

    # ── Model selection ────────────────────────────────────────────────────
    if args.only:
        models_to_run = [m for m in ALL_MODELS if m in args.only]
    elif args.skip:
        models_to_run = [m for m in ALL_MODELS if m not in args.skip]
    else:
        models_to_run = list(ALL_MODELS)

    if not models_to_run:
        logger.error("No models selected.")
        return

    # ── CLI overrides ──────────────────────────────────────────────────────
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.no_wandb:
        config["wandb"]["enabled"] = False
        logger.info("W&B logging disabled (--no-wandb)")

    ecosystem = args.ecosystem or config["training"].get("train_on", "joint")
    config["training"]["train_on"] = ecosystem

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Models: {' → '.join(MODEL_REGISTRY[m]['label'] for m in models_to_run)}")
    logger.info(f"  Ecosystem: {ecosystem}  |  Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Build DataLoaders ─────────────────────────────────────────────────
    logger.info("\nBuilding dataloaders...")

    from datasets.eco_dataset import build_dataloaders, EcoTimeSeriesDataset

    # Apply band-aware NaN fill patch
    patched_getitem = make_patched_getitem(config)
    if patched_getitem is not None:
        EcoTimeSeriesDataset.__getitem__ = patched_getitem
        logger.info("  Dataset patched: band-aware NaN fill enabled")

    processed_dir     = Path(config["data"]["processed_dir"])
    ecosystem_patches = {}
    for eco in config["ecosystems"]:
        patch_dir = processed_dir / eco / "patches"
        patch_file = patch_dir / "patches.npy"
        if patch_file.exists():
            ecosystem_patches[eco] = str(patch_dir)

            # Run diagnostic on first ecosystem
            if len(ecosystem_patches) == 1:
                logger.info(f"\n  Running patch diagnostics for {eco}...")
                report = diagnose_patches(str(patch_file), config, n_sample=500)
                if report.get("problematic_fraction", 0) > 0.10:
                    logger.warning(
                        f"\n  {'!'*60}\n"
                        f"  {report['recommendation']}\n"
                        f"  {'!'*60}\n"
                    )
        else:
            logger.warning(f"  No patches for {eco}")

    if not ecosystem_patches:
        logger.error("No patches found. Run: python scripts/01_build_tensors.py")
        return

    train_loader, val_loader = build_dataloaders(config, ecosystem_patches)
    logger.info(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # ── Train sequentially ─────────────────────────────────────────────────
    total_start = time.time()
    results     = []

    for i, model_name in enumerate(models_to_run):
        logger.info(
            f"\n\n{'='*70}\n  [{i+1}/{len(models_to_run)}] "
            f"{MODEL_REGISTRY[model_name]['label']}\n{'='*70}"
        )
        run_config = copy.deepcopy(config)
        result = train_model(model_name, run_config, train_loader, val_loader, ecosystem)
        results.append(result)

    _print_summary(results)

    sp = Path(config["outputs"]["metrics"]) / "train_all_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "metrics"} for r in results], f, indent=2)

    logger.info(f"\n  Total time: {_fmt_time(time.time() - total_start)}")
    logger.info(f"  Summary → {sp}")
    logger.info("\n  Next step: python scripts/04_generate_counterfactual.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only",      nargs="+", choices=ALL_MODELS)
    parser.add_argument("--skip",      nargs="+", choices=ALL_MODELS)
    parser.add_argument("--epochs",    type=int,  default=None)
    parser.add_argument("--ecosystem", type=str,  default=None,
                        choices=["everglades", "mississippi", "joint"])
    parser.add_argument("--no-wandb",  action="store_true", dest="no_wandb")
    main(parser.parse_args())