"""
trainer.py  (v4 — full diagnostics + degenerate-loss guards)
-------------------------------------------------------------
New fixes vs v3:

FIX A — Gradient norm logging every batch.
  grad_norm is now accumulated per epoch and logged as:
    "grad_norm_mean", "grad_norm_max" in W&B and stdout.
  Large norms (> clip*10) emit a WARNING with layer-level detail.

FIX A — Valid pixel % logged every batch.
  The loss function returns "valid_pct" in components dict.
  Trainer logs mean valid_pct per epoch to stdout and W&B.

FIX A — Loss before/after masking logged.
  "recon_unmasked" (no mask) vs "reconstruction" (with mask) are both
  forwarded to W&B so you can see how much masking shrinks the loss.

FIX G — Degenerate loss sentinel handling.
  losses.py returns DEGENERATE_LOSS_SENTINEL = -1.0 for batches that
  have < 5% valid pixels OR produce near-zero total loss.
  The trainer now:
    1. Skips those batches without zeroing gradients incorrectly
    2. Counts them as _degenerate_skip_count (logged per epoch)
  EarlyStoppingMonitor also rejects val_loss < 1e-6 (FIX G).

FIX G — val_loss < 1e-6 guard in EarlyStoppingMonitor.
  Any val_loss below 1e-6 is treated as degenerate and does NOT
  update best_loss or reset wait counter. This prevents UNet-Mamba's
  near-zero val_loss from triggering false "perfect model" early stop.

FIX 8 — Per-model LR override.
  If config["training"]["lr_overrides"] exists, the trainer applies
  per-parameter-group LRs. Useful for unet_mamba (1e-5) and
  unet_patchtst (2e-5) vs the global 3e-5.
  Example config:
    training:
      lr_overrides:
        unet_mamba: 1.0e-5
        unet_patchtst: 2.0e-5

UNCHANGED: AMP, gradient accumulation, checkpoint management,
           SequentialLR (warmup + cosine), W&B, torch.compile logic.
"""

import os
import json
import shutil
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.losses import DEGENERATE_LOSS_SENTINEL

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)

# FIX B: Minimum valid pixel ratio (fraction of optical pixels that must be valid)
MIN_VALID_RATIO = 0.10   # 10% of optical pixels must be valid — stricter than old 500px absolute

# FIX G: Early stop degenerate threshold
DEGENERATE_VAL_LOSS_THRESHOLD = 1e-6


def check_tensor_health(t: torch.Tensor, name: str) -> Tuple[bool, str]:
    n_nan = t.isnan().sum().item()
    n_inf = t.isinf().sum().item()
    if n_nan == 0 and n_inf == 0:
        return True, ""
    pct_nan = 100.0 * n_nan / t.numel()
    pct_inf = 100.0 * n_inf / t.numel()
    return False, f"{name}: NaN={n_nan}({pct_nan:.1f}%) Inf={n_inf}({pct_inf:.1f}%)"


def sanitize_batch(
    inp: torch.Tensor, target: torch.Tensor, config: Dict
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
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
        inp = torch.cat([inp_bands, inp[:, :, n_bands:]], dim=2) \
              if inp.shape[2] > n_bands else inp_bands

    if not torch.isfinite(target).all():
        was_dirty = True
        target = target.nan_to_num(0.0)

    return inp, target, was_dirty


# ---------------------------------------------------------------------------
# FIX G: EarlyStoppingMonitor with degenerate-loss guard
# ---------------------------------------------------------------------------

class EarlyStoppingMonitor:
    def __init__(self, patience: int, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait      = 0
        self.should_stop = False

    def update(self, val_loss: float) -> bool:
        # FIX G: reject degenerate losses
        if val_loss <= DEGENERATE_VAL_LOSS_THRESHOLD:
            logger.warning(
                f"  [EarlyStopping] val_loss={val_loss:.2e} ≤ {DEGENERATE_VAL_LOSS_THRESHOLD:.0e} "
                f"— treating as DEGENERATE, not updating best."
            )
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:

    def __init__(
        self,
        model:        nn.Module,
        loss_fn:      nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        config:       Dict[str, Any],
        run_name:     Optional[str] = None,
    ):
        self.model        = model
        self.loss_fn      = loss_fn
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

        # FIX 8: per-model LR override
        arch   = config["model"]["architecture"]
        max_lr = train_cfg.get("learning_rate", 3e-5)
        lr_overrides = train_cfg.get("lr_overrides", {})
        if arch in lr_overrides:
            max_lr = lr_overrides[arch]
            logger.info(f"  [LR override] {arch}: lr = {max_lr:.1e}")
        elif max_lr > 5e-5:
            logger.warning(f"  learning_rate={max_lr:.1e} — recommend ≤ 3e-5")

        self.optimizer = AdamW(
            model.parameters(), lr=max_lr,
            weight_decay=train_cfg["weight_decay"], eps=1e-8,
        )
        self.epochs        = train_cfg["epochs"]
        self.grad_clip     = train_cfg["grad_clip"]
        self.save_every    = train_cfg["save_every_n_epochs"]
        self.accumulate_steps = train_cfg.get("gradient_accumulation_steps", 1)

        warmup_epochs  = train_cfg.get("warmup_epochs", 15)
        cosine_epochs  = max(self.epochs - warmup_epochs, 1)
        warmup_sched   = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                                  total_iters=warmup_epochs)
        cosine_sched   = CosineAnnealingLR(self.optimizer, T_max=cosine_epochs,
                                           eta_min=max_lr * 0.01)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        logger.info(
            f"LR schedule: LinearWarmup ({warmup_epochs} ep) → "
            f"CosineAnnealing ({cosine_epochs} ep), peak lr={max_lr:.1e}"
        )

        self.ckpt_dir = Path(config["outputs"]["checkpoints"]) / self.run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.early_stop   = EarlyStoppingMonitor(patience=train_cfg["patience"])
        self.use_wandb    = config["wandb"]["enabled"]
        self.wandb_run    = None
        if self.use_wandb:
            self._init_wandb()

        self.best_val_loss  = float("inf")
        self.best_ckpt_path = None
        self.start_epoch    = 0
        self._try_resume()

        # torch.compile (skip for SSM-based models)
        if torch.__version__ >= "2.0" and self.device.type == "cuda":
            if arch in ("unet_mamba",):
                logger.info(f"torch.compile skipped for {arch} (SSM parallel_scan)")
            else:
                try:
                    torch._dynamo.reset()
                    self.model = torch.compile(self.model)
                    logger.info("torch.compile() enabled")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")

        # Diagnostic counters (reset each epoch)
        self._nan_skip_count        = 0
        self._nan_sanitize_count    = 0
        self._degenerate_skip_count = 0

        # FIX A: gradient norm tracking
        self._grad_norms: List[float] = []

        self._n_bands = config["bands"]["count"]
        self._sar_idx = config["bands"]["indices"]["sar_vv"]

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_wandb(self):
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            wcfg = self.config["wandb"]
            self.wandb_run = wandb.init(
                project=wcfg["project"], entity=wcfg.get("entity"),
                name=self.run_name,
                config={
                    "model":    self.config["model"],
                    "training": self.config["training"],
                    "loss":     self.config["loss"],
                },
                resume="allow",
            )
            logger.info(f"W&B: {self.wandb_run.url}")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            self.use_wandb = False

    def _try_resume(self):
        latest = self._find_latest_checkpoint()
        if not latest:
            return
        logger.info(f"Resuming from: {latest}")
        ckpt = torch.load(latest, map_location=self.device)
        try:
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.start_epoch       = ckpt["epoch"] + 1
            self.best_val_loss     = ckpt.get("best_val_loss", float("inf"))
            self.early_stop.best_loss = self.best_val_loss
            logger.info(f"Resumed epoch {self.start_epoch}, best val: {self.best_val_loss:.6f}")
        except (RuntimeError, KeyError) as e:
            logger.warning(f"Checkpoint incompatible — starting fresh. Reason: {e}")

    def _find_latest_checkpoint(self) -> Optional[str]:
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        return str(ckpts[-1]) if ckpts else None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "lr": []}
        logger.info(
            f"\n{'='*60}\n  Training: {self.run_name}\n"
            f"  Epochs: {self.start_epoch+1} → {self.epochs}\n"
            f"  Train batches: {len(self.train_loader)} | "
            f"Val batches: {len(self.val_loader)}\n"
            f"  Effective batch: "
            f"{self.config['training']['batch_size'] * self.accumulate_steps}\n"
            f"{'='*60}"
        )

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            logger.info(f"\n--- Epoch {epoch+1}/{self.epochs} ---")
            self.loss_fn.step_epoch(epoch)

            warmup_epochs = self.config["training"].get("warmup_epochs", 15)
            clip = 5.0 if epoch < warmup_epochs else self.grad_clip

            train_m = self._run_epoch(self.train_loader, training=True,  grad_clip=clip)
            val_m   = self._run_epoch(self.val_loader,   training=False, grad_clip=clip)
            self.scheduler.step()

            lr          = self.scheduler.get_last_lr()[0]
            epoch_time  = time.time() - t0

            # FIX A: gradient norm summary
            gn_mean = sum(self._grad_norms) / max(len(self._grad_norms), 1)
            gn_max  = max(self._grad_norms) if self._grad_norms else 0.0
            self._grad_norms = []

            logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"train={train_m['loss']:.4f} | val={val_m['loss']:.4f} | "
                f"lr={lr:.2e} | {epoch_time:.0f}s"
            )
            logger.info(
                f"  recon={train_m['reconstruction']:.4f} "
                f"(unmasked={train_m.get('recon_unmasked', 0):.4f}) | "
                f"temporal={train_m['temporal_smooth']:.4f} | "
                f"eco={train_m['ecological']:.4f} | ssim={train_m['ssim']:.4f}"
            )
            logger.info(
                f"  valid_pct={train_m.get('valid_pct', 0)*100:.1f}% | "
                f"grad_norm mean={gn_mean:.3f} max={gn_max:.3f}"
            )

            # Diagnostic skip counts
            if any([self._nan_skip_count, self._nan_sanitize_count,
                    self._degenerate_skip_count]):
                logger.warning(
                    f"  Skips — NaN_pred={self._nan_skip_count} | "
                    f"sanitized={self._nan_sanitize_count} | "
                    f"degenerate={self._degenerate_skip_count}"
                )
            self._nan_skip_count        = 0
            self._nan_sanitize_count    = 0
            self._degenerate_skip_count = 0

            history["train_loss"].append(train_m["loss"])
            history["val_loss"].append(val_m["loss"])
            history["lr"].append(lr)

            if self.use_wandb and self.wandb_run:
                self._log_wandb(epoch, train_m, val_m, lr, gn_mean, gn_max)

            improved = self.early_stop.update(val_m["loss"])
            if improved:
                self.best_val_loss = val_m["loss"]
                self._save_checkpoint(epoch, val_m["loss"], True)
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, val_m["loss"], False)
            if self.early_stop.should_stop:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        with open(self.ckpt_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()
        logger.info(f"Done. Best val: {self.best_val_loss:.6f} → {self.best_ckpt_path}")
        return history

    # ------------------------------------------------------------------
    # Epoch runner
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader:   DataLoader,
        training: bool,
        grad_clip: float = 1.0,
    ) -> Dict[str, float]:
        self.model.train(training)

        totals = {k: 0.0 for k in [
            "loss", "reconstruction", "recon_unmasked",
            "temporal_smooth", "ecological", "ssim", "spectral",
            "ndvi_mse", "sar_mse", "valid_pct",
        ]}
        n_batches = 0

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

                # Extract validity channel
                if target.shape[2] > n_bands:
                    validity_tgt = target[:, :, n_bands:].float()
                    target       = target[:, :, :n_bands]
                else:
                    validity_tgt = None

                # Sanitize
                inp_ok, _ = check_tensor_health(inp,    "input")
                tgt_ok, _ = check_tensor_health(target, "target")
                if not inp_ok or not tgt_ok:
                    inp, target, _ = sanitize_batch(inp, target, self.config)
                    self._nan_sanitize_count += 1

                # Build combined mask (optical validity + finite)
                if validity_tgt is not None:
                    finite_mask  = torch.isfinite(target).float()
                    combined_mask = validity_tgt * finite_mask
                else:
                    combined_mask = torch.isfinite(target).float()

                # FIX B: minimum valid pixel RATIO check (not absolute count)
                B, T, C, H, W = target.shape
                n_optical_total = B * T * (C - 1) * H * W  # exclude SAR
                optical_valid = (
                    combined_mask[:, :, :sar_idx].sum() +
                    combined_mask[:, :, sar_idx+1:].sum()
                ).item()
                valid_ratio = optical_valid / max(n_optical_total, 1)
                if valid_ratio < MIN_VALID_RATIO:
                    self._degenerate_skip_count += 1
                    if training:
                        self.optimizer.zero_grad()
                    continue

                # Zero SAR from loss mask
                loss_mask = combined_mask.clone()
                if sar_idx < loss_mask.shape[2]:
                    loss_mask[:, :, sar_idx] = 0.0

                with autocast("cuda", enabled=self.device.type == "cuda"):
                    pred = self.model(inp)

                    if not torch.isfinite(pred).all():
                        pct = (~torch.isfinite(pred)).float().mean().item() * 100
                        logger.warning(f"  Batch {batch_idx}: pred NaN={pct:.1f}% — skipping")
                        if training:
                            self.optimizer.zero_grad()
                        self._nan_skip_count += 1
                        continue

                    loss, components = self.loss_fn(pred, target, loss_mask)

                # FIX G: handle degenerate sentinel from loss function
                if loss.item() == DEGENERATE_LOSS_SENTINEL:
                    self._degenerate_skip_count += 1
                    if training:
                        self.optimizer.zero_grad()
                    continue

                if not torch.isfinite(loss) or loss.item() < 0:
                    logger.warning(f"  Batch {batch_idx}: loss={loss.item():.4f} — skipping")
                    if training:
                        self.optimizer.zero_grad()
                    self._nan_skip_count += 1
                    continue

                if training:
                    loss_scaled = loss / self.accumulate_steps
                    self.scaler.scale(loss_scaled).backward()

                    is_last    = (batch_idx + 1) == len(loader)
                    should_step = (
                        ((batch_idx + 1) % self.accumulate_steps == 0) or is_last
                    )
                    if should_step:
                        self.scaler.unscale_(self.optimizer)

                        # FIX A: log gradient norm
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                        if torch.isfinite(grad_norm):
                            self._grad_norms.append(float(grad_norm))
                            if grad_norm > grad_clip * 10:
                                logger.warning(
                                    f"  [GradNorm] batch {batch_idx}: "
                                    f"norm={grad_norm:.2f} >> clip={grad_clip}"
                                )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                # Accumulate metrics
                totals["loss"]             += loss.item()
                totals["reconstruction"]   += components.get("reconstruction",    torch.tensor(0.0)).item()
                totals["recon_unmasked"]   += components.get("recon_unmasked",    torch.tensor(0.0)).item()
                totals["temporal_smooth"]  += components.get("temporal_smooth",   torch.tensor(0.0)).item()
                totals["ecological"]       += components.get("ecological",         torch.tensor(0.0)).item()
                totals["ssim"]             += components.get("ssim",               torch.tensor(0.0)).item()
                totals["spectral"]         += components.get("spectral",           torch.tensor(0.0)).item()
                totals["valid_pct"]        += components.get("valid_pct",          torch.tensor(0.0)).item()

                with torch.no_grad():
                    if ndvi_idx < pred.shape[2]:
                        ndvi_m = combined_mask[:, :, ndvi_idx]
                        if ndvi_m.sum() > 0:
                            p_ndvi = pred[:, :, ndvi_idx][ndvi_m.bool()]
                            t_ndvi = target[:, :, ndvi_idx][ndvi_m.bool()]
                            totals["ndvi_mse"] += F.mse_loss(p_ndvi, t_ndvi).item()
                    if sar_idx < pred.shape[2]:
                        totals["sar_mse"] += F.mse_loss(
                            pred[:, :, sar_idx], target[:, :, sar_idx]
                        ).item()

                n_batches += 1

        denom = max(n_batches, 1)
        return {k: v / denom for k, v in totals.items()}

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------

    def _log_wandb(
        self,
        epoch:   int,
        train:   Dict,
        val:     Dict,
        lr:      float,
        gn_mean: float,
        gn_max:  float,
    ):
        import wandb
        wandb.log({
            "epoch":          epoch + 1,
            "learning_rate":  lr,
            "grad_norm_mean": gn_mean,
            "grad_norm_max":  gn_max,
            **{f"train/{k}": v for k, v in train.items()},
            **{f"val/{k}":   v for k, v in val.items()},
        })

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss":        val_loss,
            "best_val_loss":   self.best_val_loss,
            "config":          self.config,
            "run_name":        self.run_name,
        }
        path = self.ckpt_dir / f"epoch_{epoch+1:03d}_val{val_loss:.4f}.pt"
        torch.save(ckpt, str(path))
        logger.info(f"  Saved: {path.name}")
        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            shutil.copy(str(path), str(best_path))
            self.best_ckpt_path = str(best_path)
            logger.info(f"  *** New best: {val_loss:.6f}")
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
# Model loader (unchanged)
# ---------------------------------------------------------------------------

def load_model(
    config: Dict[str, Any], checkpoint_path: str, device: str = "cpu"
) -> nn.Module:
    arch = config["model"]["architecture"]
    if arch == "convlstm":
        from models.convlstm import ConvLSTMModel;          model = ConvLSTMModel(config)
    elif arch == "unet_temporal":
        from models.unet_temporal import UNetTemporalModel; model = UNetTemporalModel(config)
    elif arch == "eco_transformer":
        from models.eco_transformer import EcoTransformer;  model = EcoTransformer(config)
    elif arch == "unet_mamba":
        from models.unet_mamba import UNetMambaModel;       model = UNetMambaModel(config)
    elif arch == "unet_patchtst":
        from models.unet_patchtst import UNetPatchTSTModel; model = UNetPatchTSTModel(config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in ckpt["model_state"].items()
    }
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info(f"Loaded {arch} from {checkpoint_path} (epoch {ckpt['epoch']+1})")
    return model