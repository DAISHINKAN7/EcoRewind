"""
trainer.py
----------
Training loop for ECO-REWIND models.
 
Features:
  - W&B logging (loss components + band-wise MSE)
  - Checkpoint saving (local + optional Drive sync)
  - Early stopping on validation loss
  - Gradient clipping
  - Handles both ConvLSTM and UNetTemporal
"""
 
import os
import json
import shutil
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
 
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
 
logger = logging.getLogger(__name__)
 
 
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
            return True   # improved
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
            return False
 
 
class Trainer:
    """
    Manages the training loop, checkpointing, and logging.
    """
 
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
        logger.info(f"Training on: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
 
        self.model = self.model.to(self.device)
 
        # Optimizer
        train_cfg = config["training"]
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg["epochs"],
            eta_min=train_cfg["learning_rate"] * 0.01,
        )
 
        self.epochs = train_cfg["epochs"]
        self.grad_clip = train_cfg["grad_clip"]
        self.save_every = train_cfg["save_every_n_epochs"]
        self.ckpt_dir = Path(config["outputs"]["checkpoints"]) / self.run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
 
        self.early_stop = EarlyStoppingMonitor(patience=train_cfg["patience"])
 
        # W&B
        self.use_wandb = config["wandb"]["enabled"]
        self.wandb_run = None
        if self.use_wandb:
            self._init_wandb()
 
        # Checkpoint tracking
        self.best_val_loss = float("inf")
        self.best_ckpt_path = None
        self.start_epoch = 0
 
        # Try to resume from existing checkpoint
        self._try_resume()
 
    def _init_wandb(self):
        try:
            import wandb
            wandb_cfg = self.config["wandb"]
            self.wandb_run = wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),
                name=self.run_name,
                config={
                    "model": self.config["model"],
                    "training": self.config["training"],
                    "loss": self.config["loss"],
                    "patches": self.config["patches"],
                },
                resume="allow",
            )
            logger.info(f"W&B initialized: {self.wandb_run.url}")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}. Continuing without W&B.")
            self.use_wandb = False
 
    def _try_resume(self):
        latest = self._find_latest_checkpoint()
        if latest:
            logger.info(f"Resuming from checkpoint: {latest}")
            ckpt = torch.load(latest, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            self.early_stop.best_loss = self.best_val_loss
            logger.info(f"Resumed at epoch {self.start_epoch}, best val loss: {self.best_val_loss:.6f}")
 
    def _find_latest_checkpoint(self) -> Optional[str]:
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        return str(ckpts[-1]) if ckpts else None
 
    def train(self) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "lr": []}
 
        n_train_batches = len(self.train_loader)
        n_val_batches = len(self.val_loader)
        logger.info(
            f"\n{'='*60}\n"
            f"  Training started\n"
            f"  Epochs:         {self.start_epoch+1} → {self.epochs}\n"
            f"  Train batches:  {n_train_batches} per epoch\n"
            f"  Val batches:    {n_val_batches} per epoch\n"
            f"{'='*60}"
        )
 
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            logger.info(f"\n--- Epoch {epoch+1}/{self.epochs} ---")
 
            # Train
            train_metrics = self._run_epoch(self.train_loader, training=True)
            # Validate
            val_metrics = self._run_epoch(self.val_loader, training=False)
 
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
 
            # Log
            epoch_time = time.time() - t0
            logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"train={train_metrics['loss']:.4f} | "
                f"val={val_metrics['loss']:.4f} | "
                f"lr={current_lr:.2e} | "
                f"{epoch_time:.0f}s"
            )
 
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["lr"].append(current_lr)
 
            if self.use_wandb and self.wandb_run:
                import wandb
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/reconstruction": train_metrics.get("reconstruction", 0),
                    "train/temporal_smooth": train_metrics.get("temporal_smooth", 0),
                    "train/ecological": train_metrics.get("ecological", 0),
                    "val/loss": val_metrics["loss"],
                    "val/reconstruction": val_metrics.get("reconstruction", 0),
                    "val/ndvi_mse": val_metrics.get("ndvi_mse", 0),
                    "val/sar_mse": val_metrics.get("sar_mse", 0),
                    "learning_rate": current_lr,
                }
                wandb.log(log_dict)
 
            # Early stopping check
            improved = self.early_stop.update(val_metrics["loss"])
            if improved:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics["loss"], is_best=True)
 
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, val_metrics["loss"], is_best=False)
 
            if self.early_stop.should_stop:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {self.early_stop.patience} epochs)"
                )
                break
 
        # Save history
        hist_path = self.ckpt_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
 
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()
 
        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.6f}")
        logger.info(f"Best checkpoint: {self.best_ckpt_path}")
        return history
 
    def _run_epoch(
        self, loader: DataLoader, training: bool
    ) -> Dict[str, float]:
        self.model.train(training)
        ctx = torch.enable_grad() if training else torch.no_grad()
 
        total_loss = 0.0
        total_recon = 0.0
        total_temporal = 0.0
        total_eco = 0.0
        total_ndvi_mse = 0.0
        total_sar_mse = 0.0
        n_batches = 0
 
        ndvi_idx = self.config["bands"]["indices"]["ndvi"]
        sar_idx = self.config["bands"]["indices"]["sar_vv"]
 
        with ctx:
            for batch in loader:
                inp = batch["input"].to(self.device)       # (B, T_in, C, H, W)
                target = batch["target"].to(self.device)   # (B, T_out, C, H, W)
 
                # Model receives full inp (n_bands + validity channel).
                # Target is stripped to n_bands only — model predicts bands, not the mask.
                n_bands = self.config["bands"]["count"]
                if target.shape[2] > n_bands:
                    validity_tgt = target[:, :, n_bands:].float()  # (B, T, 1, H, W)
                    target = target[:, :, :n_bands]                 # (B, T, n_bands, H, W)
                else:
                    validity_tgt = None
 
                pred = self.model(inp)   # inp keeps all channels; model handles validity internally
 
                loss, components = self.loss_fn(pred, target, validity_tgt)
 
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
 
                total_loss += loss.item()
                total_recon += components["reconstruction"].item()
                total_temporal += components["temporal_smooth"].item()
                total_eco += components["ecological"].item()
 
                # Band-wise MSE for NDVI and SAR
                with torch.no_grad():
                    if ndvi_idx < pred.shape[2]:
                        total_ndvi_mse += F.mse_loss(
                            pred[:, :, ndvi_idx], target[:, :, ndvi_idx]
                        ).item()
                    if sar_idx < pred.shape[2]:
                        total_sar_mse += F.mse_loss(
                            pred[:, :, sar_idx], target[:, :, sar_idx]
                        ).item()
 
                n_batches += 1
 
        denom = max(n_batches, 1)
        return {
            "loss": total_loss / denom,
            "reconstruction": total_recon / denom,
            "temporal_smooth": total_temporal / denom,
            "ecological": total_eco / denom,
            "ndvi_mse": total_ndvi_mse / denom,
            "sar_mse": total_sar_mse / denom,
        }
 
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "run_name": self.run_name,
        }
 
        path = self.ckpt_dir / f"epoch_{epoch+1:03d}_val{val_loss:.4f}.pt"
        torch.save(ckpt, str(path))
        logger.info(f"  Checkpoint saved: {path.name}")
 
        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            shutil.copy(str(path), str(best_path))
            self.best_ckpt_path = str(best_path)
            logger.info(f"  *** New best model: {val_loss:.6f}")
 
        # Sync to Drive if enabled
        if self.config["training"].get("sync_checkpoints_to_drive", False):
            self._sync_to_drive(path)
 
        # Keep only best N checkpoints
        self._prune_checkpoints()
 
    def _prune_checkpoints(self):
        keep_n = self.config["training"].get("keep_best_n", 3)
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > keep_n:
            oldest = ckpts.pop(0)
            oldest.unlink()
 
    def _sync_to_drive(self, local_path: Path):
        drive_path = self.config["training"].get("checkpoint_drive_path", "")
        if not drive_path:
            return
        try:
            dest = Path(drive_path) / self.run_name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(local_path), str(dest / local_path.name))
            # Also copy best model
            best_local = self.ckpt_dir / "best_model.pt"
            if best_local.exists():
                shutil.copy(str(best_local), str(dest / "best_model.pt"))
            logger.info(f"  Synced to Drive: {dest / local_path.name}")
        except Exception as e:
            logger.warning(f"  Drive sync failed: {e}")
 
 
import torch.nn.functional as F
 
 
def load_model(config: Dict[str, Any], checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load a trained model from checkpoint."""
    arch = config["model"]["architecture"]
    if arch == "convlstm":
        from models.convlstm import ConvLSTMModel
        model = ConvLSTMModel(config)
    elif arch == "unet_temporal":
        from models.unet_temporal import UNetTemporalModel
        model = UNetTemporalModel(config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
 
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path} (epoch {ckpt['epoch']+1})")
    return model