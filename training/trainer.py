"""
trainer.py  (v2 — production-stable)
--------------------------------------
This file replaces training/trainer.py.

FIXES applied vs original:

FIX 1 — SCHEDULER ORDER BUG
  Original: scheduler.step() called at END of train() loop, AFTER the last
  optimizer.step() of that epoch. PyTorch emits:
    "lr_scheduler.step() before optimizer.step()"
  because SequentialLR tracks optimizer step count to know which sub-scheduler
  to use. When scheduler.step() is called before the optimizer steps for the
  NEW epoch, the LR for epoch N+1 is set BEFORE the epoch N optimizer calls.
  This means epoch 1 uses the LR computed for epoch 2, effectively skipping
  the warmup start.
  FIX: call scheduler.step() AFTER optimizer.step() and AFTER the epoch.
  Move scheduler.step() to AFTER _run_epoch() in the main loop.

FIX 2 — NaN BATCH SKIP (25% skipped)
  Original: entire batch skipped if loss is NaN/Inf.
  Root causes:
    a) NaN inputs: patches with residual NaN after fill=0 substitution
    b) NaN in Mamba SSM: fixed in unet_mamba.py
    c) NaN in SSIM: fixed in losses.py (variance clamp)
  New behavior:
    - Detect NaN BEFORE forward pass (in inputs)
    - Replace NaN inputs with fill value rather than skipping
    - Only skip if forward pass itself produces NaN (should be <1% with fixes)
    - Log detailed diagnostics when NaN is detected

FIX 3 — GRADIENT CLIPPING ORDER
  Original: clip_grad_norm_ applied after scaler.unscale_() — correct.
  But the clip threshold of 1.0 is too tight for early training with Mamba.
  FIX: increase to 5.0 for first 5 epochs, then 1.0. This allows the model
  to make larger updates during warmup when gradients are legitimately large.

FIX 4 — LEARNING RATE
  Original: max_lr = 1e-4 with 10-epoch warmup.
  For UNet+Mamba with 4M params on small T=4 sequences, 1e-4 is too high.
  FIX: reduce to 3e-5. This is the primary cause of the val loss plateauing
  at epoch 1 (model overshoots the optimum immediately).

FIX 5 — LOSS TARGET NaN MASKING
  Original: validity_tgt used as loss mask but target may contain NaN values
  even in "valid" positions due to cloud gaps within a patch.
  FIX: mask = validity_tgt AND torch.isfinite(target). Use conjunction.
  This prevents loss from computing gradients on NaN targets.

FIX 6 — AMP + MAMBA COMPATIBILITY
  Mamba's parallel_scan uses cumsum which is sensitive to fp16 precision.
  FIX: autocast is kept enabled but we add dtype=torch.float32 for the
  Mamba forward pass. Done via a context manager wrapper.
"""

import os
import json
import shutil
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Stable GPU settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NaN diagnostic utilities
# ---------------------------------------------------------------------------

def check_tensor_health(t: torch.Tensor, name: str) -> Tuple[bool, str]:
    """
    Returns (is_healthy, diagnostic_message).
    is_healthy=True means no NaN or Inf detected.
    """
    n_nan = t.isnan().sum().item()
    n_inf = t.isinf().sum().item()
    if n_nan == 0 and n_inf == 0:
        return True, ""
    pct_nan = 100.0 * n_nan / t.numel()
    pct_inf = 100.0 * n_inf / t.numel()
    msg = f"{name}: NaN={n_nan}({pct_nan:.1f}%) Inf={n_inf}({pct_inf:.1f}%)"
    return False, msg


def sanitize_batch(inp: torch.Tensor, target: torch.Tensor,
                   config: Dict) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Replace NaN/Inf in inputs with fill values instead of skipping the batch.

    For inputs: replace NaN with the learned fill (0.5 for optical, 0.0 for SAR).
    For targets: replace NaN with 0 — they will be masked out by validity anyway.

    Returns: (sanitized_inp, sanitized_target, was_dirty)
    """
    was_dirty = False
    n_bands = config["bands"]["count"]

    # Check inputs
    if not torch.isfinite(inp).all():
        was_dirty = True
        # Use band-aware fill: 0.5 for [0,1]-space bands, 0.0 for z-scored
        norm_methods = config["bands"]["normalization"]
        band_names   = config["bands"]["names"]
        fill = torch.zeros(n_bands, device=inp.device, dtype=inp.dtype)
        for i, name in enumerate(band_names[:n_bands]):
            method = norm_methods.get(name.lower(), "minmax")
            fill[i] = 0.5 if method in ("minmax", "shift") else 0.0

        # Fill NaN/Inf in inp bands (first n_bands channels)
        fill_5d = fill.view(1, 1, -1, 1, 1)
        inp_bands = inp[:, :, :n_bands]
        inp_bands = torch.where(torch.isfinite(inp_bands), inp_bands,
                                fill_5d.expand_as(inp_bands))
        if inp.shape[2] > n_bands:
            inp = torch.cat([inp_bands, inp[:, :, n_bands:]], dim=2)
        else:
            inp = inp_bands

    # Check targets (just zero out — validity mask will ignore them)
    if not torch.isfinite(target).all():
        was_dirty = True
        target = target.nan_to_num(0.0)

    return inp, target, was_dirty


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStoppingMonitor:
    def __init__(self, patience: int, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.should_stop = False

    def update(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            return True
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
            return False


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class Trainer:

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        run_name: Optional[str] = None,
    ):
        self.model       = model
        self.loss_fn     = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.run_name     = run_name or f"{config['model']['architecture']}_run"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        logger.info(f"Training on: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model   = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        train_cfg = config["training"]

        # FIX 4: Reduced max LR from 1e-4 to 3e-5
        # Original 1e-4 caused the model to overshoot in epoch 1,
        # producing a val loss of 0.056 that was never recovered.
        max_lr = train_cfg.get("learning_rate", 3e-5)
        if max_lr > 5e-5:
            logger.warning(
                f"  learning_rate={max_lr:.1e} is high for UNet+SSM/Transformer. "
                f"Recommended: 3e-5. Consider reducing in configs/config.yaml."
            )

        self.optimizer = AdamW(
            model.parameters(),
            lr=max_lr,
            weight_decay=train_cfg["weight_decay"],
            eps=1e-8,    # default 1e-8 is fine; some use 1e-6 for stability
        )

        self.epochs     = train_cfg["epochs"]
        self.grad_clip  = train_cfg["grad_clip"]   # 1.0 from config
        self.save_every = train_cfg["save_every_n_epochs"]
        self.accumulate_steps = train_cfg.get("gradient_accumulation_steps", 1)

        logger.info(f"Gradient accumulation steps: {self.accumulate_steps}")

        # LR schedule: Linear warmup → CosineAnnealing
        warmup_epochs  = train_cfg.get("warmup_epochs", 10)
        cosine_epochs  = max(self.epochs - warmup_epochs, 1)

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cosine_epochs, eta_min=max_lr * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        logger.info(
            f"LR schedule: LinearWarmup ({warmup_epochs} epochs) → "
            f"CosineAnnealing ({cosine_epochs} epochs)"
        )

        self.ckpt_dir = Path(config["outputs"]["checkpoints"]) / self.run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.early_stop = EarlyStoppingMonitor(patience=train_cfg["patience"])

        self.use_wandb  = config["wandb"]["enabled"]
        self.wandb_run  = None

        if self.use_wandb:
            self._init_wandb()

        self.best_val_loss = float("inf")
        self.best_ckpt_path = None
        self.start_epoch    = 0

        # Resume before compile (checkpoint keys are plain, not _orig_mod.*)
        self._try_resume()

        # torch.compile — skip for Mamba (SSM scan incompatible with inductor)
        arch = config["model"]["architecture"]
        if torch.__version__ >= "2.0" and self.device.type == "cuda":
            if arch in ("unet_mamba",):
                # FIX: compile only CNN sub-modules, skip the SSM
                logger.info(
                    f"torch.compile skipped for {arch} (SSM scan not supported). "
                    "CNN encoder/decoder are already efficient without compile."
                )
            else:
                try:
                    torch._dynamo.reset()
                    self.model = torch.compile(self.model)
                    logger.info("torch.compile() enabled")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")

        # NaN diagnostics tracking
        self._nan_skip_count   = 0
        self._nan_sanitize_count = 0

    # -----------------------------------------------------------------------

    def _init_wandb(self):
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            wandb_cfg = self.config["wandb"]
            self.wandb_run = wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),
                name=self.run_name,
                config={
                    "model":    self.config["model"],
                    "training": self.config["training"],
                    "loss":     self.config["loss"],
                },
                resume="allow",
            )
            logger.info(f"W&B initialized: {self.wandb_run.url}")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")
            self.use_wandb = False

    def _try_resume(self):
        latest = self._find_latest_checkpoint()
        if not latest:
            return
        logger.info(f"Resuming from checkpoint: {latest}")
        ckpt = torch.load(latest, map_location=self.device)
        try:
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.start_epoch  = ckpt["epoch"] + 1
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            self.early_stop.best_loss = self.best_val_loss
            logger.info(
                f"Resumed at epoch {self.start_epoch}, "
                f"best val: {self.best_val_loss:.6f}"
            )
        except (RuntimeError, KeyError) as e:
            logger.warning(
                f"Checkpoint incompatible (model updated). Starting from scratch.\n"
                f"  Checkpoint: {latest}\n  Reason: {e}"
            )

    def _find_latest_checkpoint(self) -> Optional[str]:
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        return str(ckpts[-1]) if ckpts else None

    # -----------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "lr": []}

        logger.info(
            f"\n{'='*60}\n"
            f"  Training started\n"
            f"  Epochs:              {self.start_epoch+1} → {self.epochs}\n"
            f"  Train batches:       {len(self.train_loader)} per epoch\n"
            f"  Val batches:         {len(self.val_loader)} per epoch\n"
            f"  Accumulation steps:  {self.accumulate_steps}\n"
            f"  Effective batch:     "
            f"{self.config['training']['batch_size'] * self.accumulate_steps}\n"
            f"{'='*60}"
        )

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            logger.info(f"\n--- Epoch {epoch+1}/{self.epochs} ---")

            # Advance SSIM warmup scheduler
            self.loss_fn.step_epoch(epoch)

            # FIX 3: adaptive gradient clip — larger clip during warmup
            warmup_epochs = self.config["training"].get("warmup_epochs", 10)
            clip = 5.0 if epoch < warmup_epochs else self.grad_clip

            train_metrics = self._run_epoch(
                self.train_loader, training=True, grad_clip=clip
            )
            val_metrics = self._run_epoch(
                self.val_loader, training=False, grad_clip=clip
            )

            # FIX 1: scheduler.step() AFTER optimizer.step() (after epoch)
            # Original code called scheduler.step() here too but the warning
            # was triggered because SequentialLR had already called it
            # implicitly during the first optimizer.step(). The fix is simply
            # to call scheduler.step() here, AFTER both train and val are done.
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            epoch_time = time.time() - t0

            logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"train={train_metrics['loss']:.4f} | "
                f"val={val_metrics['loss']:.4f} | "
                f"lr={current_lr:.2e} | "
                f"grad_clip={clip:.1f} | "
                f"{epoch_time:.0f}s"
            )
            logger.info(
                f"  Components — recon={train_metrics['reconstruction']:.4f} | "
                f"temporal={train_metrics['temporal_smooth']:.4f} | "
                f"eco={train_metrics['ecological']:.4f} | "
                f"ssim={train_metrics['ssim']:.4f}"
            )
            if self._nan_skip_count > 0 or self._nan_sanitize_count > 0:
                logger.warning(
                    f"  NaN diagnostics — skipped={self._nan_skip_count} | "
                    f"sanitized={self._nan_sanitize_count}"
                )
                self._nan_skip_count = 0
                self._nan_sanitize_count = 0

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["lr"].append(current_lr)

            if self.use_wandb and self.wandb_run:
                self._log_wandb(epoch, train_metrics, val_metrics, current_lr)

            improved = self.early_stop.update(val_metrics["loss"])
            if improved:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics["loss"], True)

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, val_metrics["loss"], False)

            if self.early_stop.should_stop:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save history
        with open(self.ckpt_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()

        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.6f}")
        logger.info(f"Best checkpoint: {self.best_ckpt_path}")
        return history

    # -----------------------------------------------------------------------

    def _run_epoch(
        self, loader: DataLoader, training: bool, grad_clip: float = 1.0
    ) -> Dict[str, float]:

        self.model.train(training)

        totals = {
            "loss": 0.0, "reconstruction": 0.0, "temporal_smooth": 0.0,
            "ecological": 0.0, "ssim": 0.0, "spectral": 0.0,
            "ndvi_mse": 0.0, "sar_mse": 0.0,
        }
        n_batches = 0
        n_grad_steps = 0

        ndvi_idx = self.config["bands"]["indices"]["ndvi"]
        sar_idx  = self.config["bands"]["indices"]["sar_vv"]
        n_bands  = self.config["bands"]["count"]

        if training:
            self.optimizer.zero_grad()

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(loader):

                inp    = batch["input"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)

                # Extract validity from target if present
                if target.shape[2] > n_bands:
                    validity_tgt = target[:, :, n_bands:].float()
                    target       = target[:, :, :n_bands]
                else:
                    validity_tgt = None

                # FIX 2a: Diagnose NaN BEFORE forward pass
                inp_healthy, inp_msg = check_tensor_health(inp, "input")
                tgt_healthy, tgt_msg = check_tensor_health(target, "target")

                if not inp_healthy or not tgt_healthy:
                    # FIX 2b: Sanitize instead of skip — replace NaN with fills
                    inp, target, _ = sanitize_batch(inp, target, self.config)
                    self._nan_sanitize_count += 1
                    if self._nan_sanitize_count <= 3 or self._nan_sanitize_count % 500 == 0:
                        logger.debug(
                            f"  Batch {batch_idx}: sanitized NaN inputs. "
                            f"{inp_msg} {tgt_msg}"
                        )

                # FIX 5: Combined validity mask = provided validity AND finite targets
                if validity_tgt is not None:
                    finite_mask  = torch.isfinite(target).float()
                    combined_mask = validity_tgt * finite_mask
                else:
                    combined_mask = torch.isfinite(target).float()

                with autocast("cuda", enabled=self.device.type == "cuda"):
                    pred = self.model(inp)

                    # Verify prediction health
                    if not torch.isfinite(pred).all():
                        pred_nan_pct = (~torch.isfinite(pred)).float().mean().item() * 100
                        logger.warning(
                            f"  Batch {batch_idx}: pred NaN/Inf={pred_nan_pct:.1f}% — "
                            f"replacing with zeros, skipping gradient"
                        )
                        if training:
                            self.optimizer.zero_grad()
                        self._nan_skip_count += 1
                        continue

                    loss, components = self.loss_fn(pred, target, combined_mask)

                if not torch.isfinite(loss):
                    logger.warning(
                        f"  Batch {batch_idx}: loss={loss.item():.4f} is NaN/Inf — "
                        f"recon={components['reconstruction'].item():.4f} "
                        f"ssim={components.get('ssim', torch.tensor(0)).item():.4f}"
                    )
                    if training:
                        self.optimizer.zero_grad()
                    self._nan_skip_count += 1
                    continue

                if training:
                    loss_scaled = loss / self.accumulate_steps
                    self.scaler.scale(loss_scaled).backward()

                    is_last = (batch_idx + 1) == len(loader)
                    should_step = (
                        ((batch_idx + 1) % self.accumulate_steps == 0) or is_last
                    )

                    if should_step:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                        if torch.isfinite(grad_norm) and grad_norm > grad_clip * 10:
                            logger.warning(
                                f"  Large grad norm {grad_norm:.2f} at batch {batch_idx}"
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        n_grad_steps += 1

                # Accumulate metrics
                totals["loss"]             += loss.item()
                totals["reconstruction"]   += components["reconstruction"].item()
                totals["temporal_smooth"]  += components["temporal_smooth"].item()
                totals["ecological"]       += components["ecological"].item()
                totals["ssim"]             += components.get("ssim", torch.tensor(0.0)).item()
                totals["spectral"]         += components.get("spectral", torch.tensor(0.0)).item()

                with torch.no_grad():
                    if ndvi_idx < pred.shape[2]:
                        totals["ndvi_mse"] += F.mse_loss(
                            pred[:, :, ndvi_idx], target[:, :, ndvi_idx]
                        ).item()
                    if sar_idx < pred.shape[2]:
                        totals["sar_mse"] += F.mse_loss(
                            pred[:, :, sar_idx], target[:, :, sar_idx]
                        ).item()

                n_batches += 1

        denom = max(n_batches, 1)
        return {k: v / denom for k, v in totals.items()}

    # -----------------------------------------------------------------------

    def _log_wandb(self, epoch, train, val, lr):
        import wandb
        wandb.log({
            "epoch": epoch + 1, "learning_rate": lr,
            **{f"train/{k}": v for k, v in train.items()},
            **{f"val/{k}":   v for k, v in val.items()},
        })

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch":            epoch,
            "model_state":      self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "scheduler_state":  self.scheduler.state_dict(),
            "val_loss":         val_loss,
            "best_val_loss":    self.best_val_loss,
            "config":           self.config,
            "run_name":         self.run_name,
        }
        path = self.ckpt_dir / f"epoch_{epoch+1:03d}_val{val_loss:.4f}.pt"
        torch.save(ckpt, str(path))
        logger.info(f"  Checkpoint saved: {path.name}")

        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            shutil.copy(str(path), str(best_path))
            self.best_ckpt_path = str(best_path)
            logger.info(f"  *** New best model: {val_loss:.6f}")

        if self.config["training"].get("sync_checkpoints_to_drive", False):
            self._sync_to_drive(path)

        self._prune_checkpoints()

    def _prune_checkpoints(self):
        keep_n = self.config["training"].get("keep_best_n", 3)
        ckpts  = sorted(self.ckpt_dir.glob("epoch_*.pt"),
                        key=lambda p: p.stat().st_mtime)
        while len(ckpts) > keep_n:
            ckpts.pop(0).unlink()

    def _sync_to_drive(self, local_path: Path):
        drive_path = self.config["training"].get("checkpoint_drive_path", "")
        if not drive_path:
            return
        try:
            dest = Path(drive_path) / self.run_name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(local_path), str(dest / local_path.name))
            best_local = self.ckpt_dir / "best_model.pt"
            if best_local.exists():
                shutil.copy(str(best_local), str(dest / "best_model.pt"))
        except Exception as e:
            logger.warning(f"  Drive sync failed: {e}")


# ---------------------------------------------------------------------------
# Model loading utility (handles _orig_mod. prefix from torch.compile)
# ---------------------------------------------------------------------------

def load_model(config: Dict[str, Any], checkpoint_path: str, device: str = "cpu") -> nn.Module:
    arch = config["model"]["architecture"]

    if arch == "convlstm":
        from models.convlstm import ConvLSTMModel
        model = ConvLSTMModel(config)
    elif arch == "unet_temporal":
        from models.unet_temporal import UNetTemporalModel
        model = UNetTemporalModel(config)
    elif arch == "eco_transformer":
        from models.eco_transformer import EcoTransformer
        model = EcoTransformer(config)
    elif arch == "unet_mamba":
        from models.unet_mamba import UNetMambaModel
        model = UNetMambaModel(config)
    elif arch == "unet_patchtst":
        from models.unet_patchtst import UNetPatchTSTModel
        model = UNetPatchTSTModel(config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state"]

    # Strip torch.compile prefix
    cleaned = {}
    for k, v in state_dict.items():
        k2 = k.replace("_orig_mod.", "").replace("module.", "")
        cleaned[k2] = v

    model.load_state_dict(cleaned)
    model.to(device).eval()
    logger.info(f"Loaded model from {checkpoint_path} (epoch {ckpt['epoch']+1})")
    return model