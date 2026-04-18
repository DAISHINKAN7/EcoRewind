"""
config_patch.py
---------------
Drop-in config overrides for the production-stable training setup.

Usage:
    import yaml
    from config_patch import apply_stability_patch
    
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    config = apply_stability_patch(config)
    # → use config as usual

Why each change is made is documented inline.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def apply_stability_patch(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply production-stability overrides to a loaded config dict.
    Only modifies values that are known causes of instability.
    All changes are logged so the user can see exactly what changed.
    """
    changes = []

    # ── LEARNING RATE ───────────────────────────────────────────────────────
    # PROBLEM: Original 1e-4 overshoots the optimum in epoch 1.
    #   Evidence: val_loss drops to 0.0568 in epoch 1, then INCREASES to 0.1059
    #   by epoch 10 and stays stuck. This is a classic overshoot pattern.
    #   With LR=1e-4 and T=4 sequences, the optimizer takes one large step that
    #   lands far from the basin, then can't descend back because cosine LR
    #   never rises above 1e-4.
    # FIX: 3e-5 is the empirically stable range for UNet+SSM on small T.
    #   With 10-epoch linear warmup: starts at 3e-6, peaks at 3e-5. 
    #   This gives the model time to find a stable basin before full LR.
    old_lr = config["training"]["learning_rate"]
    if old_lr > 5e-5:
        config["training"]["learning_rate"] = 3e-5
        changes.append(f"  learning_rate: {old_lr:.1e} → 3e-5 (prevents epoch-1 overshoot)")

    # ── WARMUP EPOCHS ──────────────────────────────────────────────────────
    # PROBLEM: 10-epoch warmup with 1e-4 target was still reaching 5.5e-5 at epoch 5,
    #   which is too high for a model that hasn't learned basic structure yet.
    # FIX: Increase warmup to 15 epochs. With 3e-5 max, this means:
    #   epoch 1: lr=3e-6, epoch 5: lr=1.2e-5, epoch 15: lr=3e-5
    #   Slow enough that the model doesn't overshoot.
    old_warmup = config["training"].get("warmup_epochs", 10)
    if old_warmup < 15:
        config["training"]["warmup_epochs"] = 15
        changes.append(f"  warmup_epochs: {old_warmup} → 15 (gentler early training)")

    # ── GRADIENT CLIPPING ──────────────────────────────────────────────────
    # PROBLEM: clip=1.0 is appropriate for post-warmup training but too tight
    #   during warmup when gradients are legitimately large (fresh random init).
    #   This wasn't the primary problem but could slow convergence.
    # NOTE: The trainer.py applies adaptive clipping: 5.0 during warmup, then
    #   falls back to grad_clip from config. So this value controls post-warmup.
    old_clip = config["training"]["grad_clip"]
    if old_clip > 2.0:
        config["training"]["grad_clip"] = 1.0
        changes.append(f"  grad_clip: {old_clip} → 1.0 (post-warmup clip)")
    # If clip was already 1.0, no change needed.

    # ── PATIENCE ───────────────────────────────────────────────────────────
    # PROBLEM: patience=25 with a plateaued val loss means training runs for
    #   25 wasted epochs. With the LR fix, the model should converge by epoch
    #   50-80, so keeping patience=25 is reasonable.
    # No change needed — patience=25 is fine.

    # ── SSIM WARMUP ────────────────────────────────────────────────────────
    # PROBLEM: Original ssim_warmup_epochs=5 turns on SSIM too early.
    #   In epoch 5, the model is still fitting basic brightness; SSIM on random
    #   predictions adds ~0.5 loss that drowns the reconstruction signal.
    # FIX: Delay SSIM to epoch 20 so reconstruction loss has time to dominate.
    old_ssim_warmup = config["loss"].get("ssim_warmup_epochs", 5)
    if old_ssim_warmup < 20:
        config["loss"]["ssim_warmup_epochs"] = 20
        changes.append(
            f"  ssim_warmup_epochs: {old_ssim_warmup} → 20 "
            "(prevents SSIM from overwhelming recon early)"
        )

    # ── LAMBDA TEMPORAL ────────────────────────────────────────────────────
    # PROBLEM: lambda_temporal=0.15 is large when temporal loss itself is ~0.04,
    #   contributing 0.006 to total loss. With negative R², the model hasn't
    #   learned the basic signal yet, so temporal consistency is a distraction.
    # FIX: Reduce to 0.05 for first phase; the loss function can be tuned later.
    old_lt = config["loss"]["lambda_temporal"]
    if old_lt > 0.1:
        config["loss"]["lambda_temporal"] = 0.05
        changes.append(
            f"  lambda_temporal: {old_lt} → 0.05 "
            "(focus on reconstruction first)"
        )

    # ── WINDOW MODE ────────────────────────────────────────────────────────
    # Ensure window_mode="all" (correct, already in config)
    # "all" gives 21 windows vs 1 with "pre_only" → 21× more training signal
    # No change needed.

    # ── SUMMARY ────────────────────────────────────────────────────────────
    if changes:
        logger.info("[config_patch] Applied stability overrides:")
        for c in changes:
            logger.info(c)
    else:
        logger.info("[config_patch] No overrides needed — config already stable.")

    return config


# ---------------------------------------------------------------------------
# Recommended configs/config.yaml training block (for reference)
# ---------------------------------------------------------------------------

RECOMMENDED_TRAINING_CONFIG = """
# Paste this into configs/config.yaml under training:
training:
  epochs: 150
  batch_size: 4
  learning_rate: 0.00003        # was 0.0001 — CRITICAL FIX
  weight_decay: 0.00001
  patience: 25
  grad_clip: 1.0

  warmup_epochs: 15             # was 10
  gradient_accumulation_steps: 4

  use_augmentation: true
  train_on: joint
  window_mode: "all"
  balanced_sampling: true

  save_every_n_epochs: 5
  keep_best_n: 3

loss:
  lambda_temporal: 0.05         # was 0.15
  lambda_ecological: 0.05
  lambda_ssim: 0.10
  lambda_spectral: 0.00
  ssim_warmup_epochs: 20        # was 5
"""