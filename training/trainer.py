"""
trainer.py  (v3 — fixes validity mask causing 0.0 loss and degenerate training)
--------------------------------------------------------------------------------

NEW FIXES vs v2:

FIX A — VALIDITY MASK: optical-only, not all-band
  Original v2 combined_mask = validity_tgt * finite_mask
  validity_tgt is the 8th patch channel = 1 only where ALL bands finite
  SAR is 87.5% NaN → validity is 87.5% zero → loss on 12.5% pixels only
  This caused UNet-Mamba to report best_val=0.0000 (loss numerically zero
  because n_valid is so small it rounds to near-zero)
  
  FIX: The new patch_sampler.py saves optical-only validity (bands 0:6).
  The trainer now correctly uses this: validity=1 where optical data is valid.
  This gives ~90% valid pixels instead of ~12%.

FIX B — MINIMUM PIXEL GUARD
  When n_valid is very small (degenerate batch), loss = tiny/tiny = 0.0
  This fools early_stop into thinking val_loss=0.0 = "perfect model"
  FIX: Require at least min_valid_pixels per batch or skip the batch.

FIX C — SAR EXCLUDED FROM PRIMARY LOSS
  SAR has only 12.5% valid pixels. Training the reconstruction loss on SAR
  with 87.5% fill values biases the model to predict the fill value.
  FIX: Pass a per-band loss weight that zeroes out SAR reconstruction.
  SAR R² will remain poor because SAR data barely exists, but this stops
  SAR from poisoning the optical band gradients.

FIX D — EVALUATOR USES OPTICAL VALIDITY ONLY
  The evaluator computes R² per band using torch.isfinite(target).
  Since SAR targets have 87.5% fill=0.0 (FINITE but wrong), the evaluator
  computes R² against those wrong values → catastrophically bad R².
  FIX: Use the validity channel from the batch to mask evaluation too.
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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)

MIN_VALID_PIXELS = 500   # Skip batch if fewer than this many valid pixels


def check_tensor_health(t: torch.Tensor, name: str) -> Tuple[bool, str]:
    n_nan = t.isnan().sum().item()
    n_inf = t.isinf().sum().item()
    if n_nan == 0 and n_inf == 0:
        return True, ""
    pct_nan = 100.0 * n_nan / t.numel()
    pct_inf = 100.0 * n_inf / t.numel()
    return False, f"{name}: NaN={n_nan}({pct_nan:.1f}%) Inf={n_inf}({pct_inf:.1f}%)"


def sanitize_batch(inp: torch.Tensor, target: torch.Tensor,
                   config: Dict) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    was_dirty = False
    n_bands = config["bands"]["count"]
    norm_methods = config["bands"]["normalization"]
    band_names = config["bands"]["names"]

    if not torch.isfinite(inp).all():
        was_dirty = True
        fill = torch.zeros(n_bands, device=inp.device, dtype=inp.dtype)
        for i, name in enumerate(band_names[:n_bands]):
            method = norm_methods.get(name.lower(), "minmax")
            fill[i] = 0.5 if method in ("minmax", "shift") else 0.0
        fill_5d = fill.view(1, 1, -1, 1, 1)
        inp_bands = inp[:, :, :n_bands]
        inp_bands = torch.where(torch.isfinite(inp_bands), inp_bands,
                                fill_5d.expand_as(inp_bands))
        inp = torch.cat([inp_bands, inp[:, :, n_bands:]], dim=2) if inp.shape[2] > n_bands else inp_bands

    if not torch.isfinite(target).all():
        was_dirty = True
        target = target.nan_to_num(0.0)

    return inp, target, was_dirty


class EarlyStoppingMonitor:
    def __init__(self, patience: int, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.should_stop = False

    def update(self, val_loss: float) -> bool:
        # FIX B: Ignore suspiciously perfect losses
        if val_loss <= 0.0:
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            return True
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
            return False


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
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_name = run_name or f"{config['model']['architecture']}_run"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        logger.info(f"Training on: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        train_cfg = config["training"]
        max_lr = train_cfg.get("learning_rate", 3e-5)
        if max_lr > 5e-5:
            logger.warning(f"  learning_rate={max_lr:.1e} — recommend 3e-5 for stability")

        self.optimizer = AdamW(
            model.parameters(), lr=max_lr,
            weight_decay=train_cfg["weight_decay"], eps=1e-8,
        )
        self.epochs = train_cfg["epochs"]
        self.grad_clip = train_cfg["grad_clip"]
        self.save_every = train_cfg["save_every_n_epochs"]
        self.accumulate_steps = train_cfg.get("gradient_accumulation_steps", 1)

        logger.info(f"Gradient accumulation steps: {self.accumulate_steps}")

        warmup_epochs = train_cfg.get("warmup_epochs", 15)
        cosine_epochs = max(self.epochs - warmup_epochs, 1)
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                                    total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_epochs,
                                              eta_min=max_lr * 0.01)
        self.scheduler = SequentialLR(self.optimizer,
                                       schedulers=[warmup_scheduler, cosine_scheduler],
                                       milestones=[warmup_epochs])
        logger.info(f"LR schedule: LinearWarmup ({warmup_epochs}) → CosineAnnealing ({cosine_epochs})")

        self.ckpt_dir = Path(config["outputs"]["checkpoints"]) / self.run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.early_stop = EarlyStoppingMonitor(patience=train_cfg["patience"])
        self.use_wandb = config["wandb"]["enabled"]
        self.wandb_run = None
        if self.use_wandb:
            self._init_wandb()

        self.best_val_loss = float("inf")
        self.best_ckpt_path = None
        self.start_epoch = 0
        self._try_resume()

        arch = config["model"]["architecture"]
        if torch.__version__ >= "2.0" and self.device.type == "cuda":
            if arch in ("unet_mamba",):
                logger.info(f"torch.compile skipped for {arch} (SSM scan incompatible with inductor)")
            else:
                try:
                    torch._dynamo.reset()
                    self.model = torch.compile(self.model)
                    logger.info("torch.compile() enabled")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")

        self._nan_skip_count = 0
        self._nan_sanitize_count = 0
        self._degenerate_skip_count = 0

        # FIX C: Build per-band loss weights that zero out SAR
        # SAR has 87.5% fill values — training on it corrupts optical gradients
        n_bands = config["bands"]["count"]
        sar_idx = config["bands"]["indices"]["sar_vv"]
        self._sar_idx = sar_idx
        self._n_bands = n_bands

    def _init_wandb(self):
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            wandb_cfg = self.config["wandb"]
            self.wandb_run = wandb.init(
                project=wandb_cfg["project"], entity=wandb_cfg.get("entity"),
                name=self.run_name,
                config={"model": self.config["model"], "training": self.config["training"],
                        "loss": self.config["loss"]},
                resume="allow",
            )
            logger.info(f"W&B initialized: {self.wandb_run.url}")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
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
            self.start_epoch = ckpt["epoch"] + 1
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            self.early_stop.best_loss = self.best_val_loss
            logger.info(f"Resumed at epoch {self.start_epoch}, best val: {self.best_val_loss:.6f}")
        except (RuntimeError, KeyError) as e:
            logger.warning(f"Checkpoint incompatible. Starting from scratch.\n  Reason: {e}")

    def _find_latest_checkpoint(self) -> Optional[str]:
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        return str(ckpts[-1]) if ckpts else None

    def train(self) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "lr": []}
        logger.info(
            f"\n{'='*60}\n  Training started\n"
            f"  Epochs: {self.start_epoch+1} → {self.epochs}\n"
            f"  Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}\n"
            f"  Effective batch: {self.config['training']['batch_size'] * self.accumulate_steps}\n"
            f"{'='*60}"
        )

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            logger.info(f"\n--- Epoch {epoch+1}/{self.epochs} ---")
            self.loss_fn.step_epoch(epoch)

            warmup_epochs = self.config["training"].get("warmup_epochs", 15)
            clip = 5.0 if epoch < warmup_epochs else self.grad_clip

            train_metrics = self._run_epoch(self.train_loader, training=True, grad_clip=clip)
            val_metrics = self._run_epoch(self.val_loader, training=False, grad_clip=clip)
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            epoch_time = time.time() - t0

            logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"train={train_metrics['loss']:.4f} | val={val_metrics['loss']:.4f} | "
                f"lr={current_lr:.2e} | {epoch_time:.0f}s"
            )
            logger.info(
                f"  Components — recon={train_metrics['reconstruction']:.4f} | "
                f"temporal={train_metrics['temporal_smooth']:.4f} | "
                f"eco={train_metrics['ecological']:.4f} | ssim={train_metrics['ssim']:.4f}"
            )

            if self._nan_skip_count > 0 or self._nan_sanitize_count > 0 or self._degenerate_skip_count > 0:
                logger.warning(
                    f"  Diagnostics — NaN skipped={self._nan_skip_count} | "
                    f"sanitized={self._nan_sanitize_count} | "
                    f"degenerate skipped={self._degenerate_skip_count}"
                )
                self._nan_skip_count = 0
                self._nan_sanitize_count = 0
                self._degenerate_skip_count = 0

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

        with open(self.ckpt_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()

        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.6f}")
        logger.info(f"Best checkpoint: {self.best_ckpt_path}")
        return history

    def _run_epoch(self, loader: DataLoader, training: bool,
                   grad_clip: float = 1.0) -> Dict[str, float]:
        self.model.train(training)
        totals = {k: 0.0 for k in ["loss", "reconstruction", "temporal_smooth",
                                     "ecological", "ssim", "spectral", "ndvi_mse", "sar_mse"]}
        n_batches = 0

        ndvi_idx = self.config["bands"]["indices"]["ndvi"]
        sar_idx = self.config["bands"]["indices"]["sar_vv"]
        n_bands = self.config["bands"]["count"]

        if training:
            self.optimizer.zero_grad()

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(loader):
                inp = batch["input"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)

                # FIX A: Extract validity channel (optical-only validity from new patch_sampler)
                if target.shape[2] > n_bands:
                    validity_tgt = target[:, :, n_bands:].float()   # (B, T, 1, H, W)
                    target = target[:, :, :n_bands]
                else:
                    validity_tgt = None

                # Sanitize NaN inputs
                inp_healthy, inp_msg = check_tensor_health(inp, "input")
                tgt_healthy, tgt_msg = check_tensor_health(target, "target")
                if not inp_healthy or not tgt_healthy:
                    inp, target, _ = sanitize_batch(inp, target, self.config)
                    self._nan_sanitize_count += 1

                # FIX A: Combined mask = optical validity (NOT all-band validity)
                # validity_tgt is now optical-only (90% valid vs old 12% valid)
                if validity_tgt is not None:
                    # Also mask where target itself is NaN (residual NaN after sanitize)
                    finite_mask = torch.isfinite(target).float()
                    combined_mask = validity_tgt * finite_mask
                else:
                    combined_mask = torch.isfinite(target).float()

                # FIX B: Skip batch if too few valid pixels (prevents 0.0 loss artifact)
                n_valid_pixels = combined_mask.sum().item()
                if n_valid_pixels < MIN_VALID_PIXELS:
                    self._degenerate_skip_count += 1
                    if training:
                        self.optimizer.zero_grad()
                    continue

                # FIX C: Zero out SAR from validity mask for LOSS computation only
                # SAR has 87.5% fill values that corrupt gradients for optical bands
                # We still predict SAR (the model outputs it) but don't train on it
                loss_mask = combined_mask.clone()
                if sar_idx < loss_mask.shape[2]:
                    loss_mask[:, :, sar_idx] = 0.0   # exclude SAR from loss

                with autocast("cuda", enabled=self.device.type == "cuda"):
                    pred = self.model(inp)

                    if not torch.isfinite(pred).all():
                        pred_nan_pct = (~torch.isfinite(pred)).float().mean().item() * 100
                        logger.warning(f"  Batch {batch_idx}: pred NaN={pred_nan_pct:.1f}% — skipping")
                        if training:
                            self.optimizer.zero_grad()
                        self._nan_skip_count += 1
                        continue

                    loss, components = self.loss_fn(pred, target, loss_mask)

                if not torch.isfinite(loss) or loss.item() <= 0.0:
                    if not torch.isfinite(loss):
                        logger.warning(f"  Batch {batch_idx}: loss=NaN/Inf — skipping")
                    if training:
                        self.optimizer.zero_grad()
                    self._nan_skip_count += 1
                    continue

                if training:
                    loss_scaled = loss / self.accumulate_steps
                    self.scaler.scale(loss_scaled).backward()

                    is_last = (batch_idx + 1) == len(loader)
                    should_step = (((batch_idx + 1) % self.accumulate_steps == 0) or is_last)
                    if should_step:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                        if torch.isfinite(grad_norm) and grad_norm > grad_clip * 10:
                            logger.warning(f"  Large grad norm {grad_norm:.2f} at batch {batch_idx}")
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                totals["loss"] += loss.item()
                totals["reconstruction"] += components["reconstruction"].item()
                totals["temporal_smooth"] += components["temporal_smooth"].item()
                totals["ecological"] += components["ecological"].item()
                totals["ssim"] += components.get("ssim", torch.tensor(0.0)).item()
                totals["spectral"] += components.get("spectral", torch.tensor(0.0)).item()

                with torch.no_grad():
                    if ndvi_idx < pred.shape[2]:
                        # FIX D: Only compute NDVI MSE where optically valid
                        ndvi_mask = combined_mask[:, :, ndvi_idx] if combined_mask.shape[2] > ndvi_idx else None
                        if ndvi_mask is not None and ndvi_mask.sum() > 0:
                            p_ndvi = pred[:, :, ndvi_idx][ndvi_mask.bool()]
                            t_ndvi = target[:, :, ndvi_idx][ndvi_mask.bool()]
                            totals["ndvi_mse"] += F.mse_loss(p_ndvi, t_ndvi).item()
                    if sar_idx < pred.shape[2]:
                        totals["sar_mse"] += F.mse_loss(
                            pred[:, :, sar_idx], target[:, :, sar_idx]).item()

                n_batches += 1

        denom = max(n_batches, 1)
        return {k: v / denom for k, v in totals.items()}

    def _log_wandb(self, epoch, train, val, lr):
        import wandb
        wandb.log({
            "epoch": epoch + 1, "learning_rate": lr,
            **{f"train/{k}": v for k, v in train.items()},
            **{f"val/{k}": v for k, v in val.items()},
        })

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch, "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss": val_loss, "best_val_loss": self.best_val_loss,
            "config": self.config, "run_name": self.run_name,
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
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
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


def load_model(config: Dict[str, Any], checkpoint_path: str, device: str = "cpu") -> nn.Module:
    arch = config["model"]["architecture"]
    if arch == "convlstm":
        from models.convlstm import ConvLSTMModel; model = ConvLSTMModel(config)
    elif arch == "unet_temporal":
        from models.unet_temporal import UNetTemporalModel; model = UNetTemporalModel(config)
    elif arch == "eco_transformer":
        from models.eco_transformer import EcoTransformer; model = EcoTransformer(config)
    elif arch == "unet_mamba":
        from models.unet_mamba import UNetMambaModel; model = UNetMambaModel(config)
    elif arch == "unet_patchtst":
        from models.unet_patchtst import UNetPatchTSTModel; model = UNetPatchTSTModel(config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state"]
    cleaned = {k.replace("_orig_mod.", "").replace("module.", ""): v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.to(device).eval()
    logger.info(f"Loaded model from {checkpoint_path} (epoch {ckpt['epoch']+1})")
    return model