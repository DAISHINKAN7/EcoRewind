"""
counterfactual.py
-----------------
Counterfactual trajectory generation.
 
Given a trained model and pre-disturbance context, rolls the model
forward past the hurricane event WITHOUT forcing hurricane conditions.
 
The resulting trajectory represents "what would have happened without
the hurricane" — compared to the actual post-hurricane observations
to quantify ecological damage.
 
MC Dropout support: run N stochastic forward passes with Dropout active
to get mean + std counterfactual trajectory (proper model uncertainty).
 
Seasonal Climatology Floor: clamp autoregressive predictions to
mean - n_sigma * std of pre-event seasonal statistics to prevent
compounding winter-dip drift across rollout iterations.
 
Output: a (T_post, C, H, W) counterfactual trajectory tensor
"""
 
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
 
logger = logging.getLogger(__name__)
 
 
class CounterfactualGenerator:
    """
    Generates counterfactual trajectories for an entire ecosystem.
 
    Process:
      1. Load the full (T, C, H, W) tensor
      2. Normalize with the pre-disturbance normalizer
      3. For each spatial patch:
           - Feed [t_event-T_in : t_event] as input context
           - Roll model forward T_post steps autoregressively
           - Optionally repeat N times with MC Dropout for uncertainty
      4. Stitch patch predictions back into full-resolution maps
      5. Inverse-normalize to physical units
    """
 
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        normalizer,
        climatology=None,
        device: str = "cpu",
    ):
        self.model = model
        self.config = config
        self.normalizer = normalizer
        self.climatology = climatology   # SeasonalClimatology instance or None
        self.device = device
        self.model.eval()
 
        self.patch_size = config["patches"]["size"]
        self.stride = config["patches"]["stride"]
        self.n_bands = config["bands"]["count"]
        self.use_validity = config["patches"]["use_validity_mask"]
        self.t_input = config["model"]["t_input"]
 
        # MC Dropout settings
        inf_cfg = config.get("inference", {})
        self.mc_samples = inf_cfg.get("mc_dropout_samples", 1)
        self.use_climatology = inf_cfg.get("climatology_anchor", False) and (climatology is not None)
        self.climatology_sigma = inf_cfg.get("climatology_sigma", 2.0)
 
        if self.mc_samples > 1:
            logger.info(
                f"MC Dropout enabled: {self.mc_samples} samples per patch. "
                "Output includes mean + std uncertainty maps."
            )
        if self.use_climatology:
            logger.info(
                f"Seasonal climatology floor active (n_sigma={self.climatology_sigma})"
            )
 
    # ------------------------------------------------------------------
 
    def generate(
        self,
        ecosystem: str,
        tensor: np.ndarray,
        validity: np.ndarray,
        n_forecast: int,
    ) -> Dict[str, np.ndarray]:
        """
        Generate counterfactual maps for the full ecosystem.
 
        Args:
            ecosystem  : "everglades" or "mississippi"
            tensor     : (T, C, H, W) raw (unnormalized) tensor
            validity   : (T, 1, H, W) validity mask
            n_forecast : number of quarters to forecast past t_event
 
        Returns dict with keys:
            "counterfactual" : (n_forecast, C, H, W) CF trajectory (physical units)
            "actual_post"    : (n_forecast, C, H, W) actual post-event trajectory
            "delta"          : (n_forecast, C, H, W) CF − actual (damage map)
            "uncertainty"    : (n_forecast, C, H, W) std of MC samples (physical units)
                               Only present when mc_dropout_samples > 1.
        """
        eco_cfg = self.config["ecosystems"][ecosystem.lower()]
        t_event = eco_cfg["t_event"]
        start_quarter = eco_cfg.get("start_quarter", 1)
        T, C, H, W = tensor.shape
 
        # Normalize full tensor (applying pre-disturbance fitted normalizer)
        logger.info(f"[{ecosystem}] Normalizing tensor for CF generation...")
        norm_tensor = self.normalizer.transform(tensor)  # (T, C, H, W)
 
        # Input context: [t_event - T_in : t_event]
        t_ctx_start = max(0, t_event - self.t_input)
        context = norm_tensor[t_ctx_start:t_event]      # (T_in, C, H, W)
        T_in_actual = context.shape[0]
 
        # Pad if context is shorter than t_input (near start of series)
        if T_in_actual < self.t_input:
            pad = np.zeros(
                (self.t_input - T_in_actual, C, H, W), dtype=np.float32
            )
            context = np.concatenate([pad, context], axis=0)
 
        # Actual post-event frames (for delta computation)
        t_post_end = min(T, t_event + n_forecast)
        actual_post_norm = norm_tensor[t_event:t_post_end]   # (n_actual, C, H, W)
        n_actual = actual_post_norm.shape[0]
 
        # Precompute climatology floors for each forecast step (in normalised space)
        clim_floors = None
        if self.use_climatology and self.climatology is not None:
            clim_floors = []
            for step in range(n_forecast):
                # quarter of the forecast step (0-indexed)
                q = ((start_quarter - 1) + t_event + step) % 4
                floor = self.climatology.get_floor(q, n_sigma=self.climatology_sigma)
                clim_floors.append(floor)   # (C, H, W)
 
        # Build counterfactual via patch stitching
        logger.info(
            f"[{ecosystem}] Generating {n_forecast} CF quarters by patch stitching "
            f"(mc_samples={self.mc_samples})..."
        )
        cf_norm, cf_std_norm = self._stitch_patches(
            context, H, W, n_forecast, clim_floors
        )  # both (n_forecast, C, H, W)
 
        # Trim to actual available frames
        n_out = min(n_forecast, n_actual)
        cf_norm = cf_norm[:n_out]
        cf_std_norm = cf_std_norm[:n_out]
        actual_post_norm = actual_post_norm[:n_out]
 
        # Inverse-normalize to physical units
        logger.info(f"[{ecosystem}] Inverse-normalizing to physical units...")
        cf_physical = np.stack([
            self.normalizer.inverse_transform(cf_norm[t]) for t in range(n_out)
        ])
        cf_std_physical = np.stack([
            self.normalizer.inverse_transform(cf_std_norm[t]) for t in range(n_out)
        ]) if self.mc_samples > 1 else None
        actual_physical = np.stack([
            self.normalizer.inverse_transform(actual_post_norm[t]) for t in range(n_out)
        ])
 
        # Compute delta: CF − actual (positive = CF higher = vegetation loss in actual)
        delta = cf_physical - actual_physical
 
        # Log basic stats
        ndvi_idx = self.config["bands"]["indices"]["ndvi"]
        ndvi_delta = delta[:, ndvi_idx]
        finite = ndvi_delta[np.isfinite(ndvi_delta)]
        if len(finite) > 0:
            logger.info(
                f"[{ecosystem}] ΔNDVI stats:\n"
                f"  Mean:   {np.mean(finite):.4f}\n"
                f"  Max:    {np.max(finite):.4f}\n"
                f"  Min:    {np.min(finite):.4f}\n"
                f"  >0.05:  {np.mean(finite > 0.05)*100:.1f}% of pixels"
            )
 
        result = {
            "counterfactual": cf_physical,
            "actual_post": actual_physical,
            "delta": delta,
        }
        if cf_std_physical is not None:
            result["uncertainty"] = cf_std_physical
 
        return result
 
    # ------------------------------------------------------------------
 
    def _stitch_patches(
        self,
        context: np.ndarray,
        H: int, W: int,
        n_forecast: int,
        clim_floors,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide over spatial domain, run model on each patch, and average
        overlapping predictions (Gaussian weighting for smooth boundaries).
 
        When mc_samples > 1, runs each patch N times with MC Dropout active
        and returns mean and std across samples.
 
        context     : (T_in, C, H, W) normalized input
        clim_floors : list of (C, H, W) or None
 
        Returns : (n_forecast, C, H, W) mean, (n_forecast, C, H, W) std
        """
        p = self.patch_size
        s = self.stride
        T_in, C = context.shape[:2]
        N = self.mc_samples
 
        # Accumulator tensors (using Welford online mean/variance for memory)
        pred_sum    = np.zeros((n_forecast, C, H, W), dtype=np.float64)
        pred_sq_sum = np.zeros((n_forecast, C, H, W), dtype=np.float64)
        weight_sum  = np.zeros((n_forecast, 1, H, W), dtype=np.float64)
 
        # Gaussian 2D weight for smooth blending
        weight_patch = self._gaussian_patch_weight(p)  # (p, p)
 
        row_starts = list(range(0, H - p + 1, s))
        col_starts = list(range(0, W - p + 1, s))
        if row_starts[-1] != H - p:
            row_starts.append(H - p)
        if col_starts[-1] != W - p:
            col_starts.append(W - p)
 
        n_patches = len(row_starts) * len(col_starts)
        logger.info(f"  Stitching {n_patches} patches × {N} MC samples...")
 
        # Enable MC Dropout if requested
        if N > 1 and hasattr(self.model, "enable_mc_dropout"):
            self.model.enable_mc_dropout()
            logger.info("  MC Dropout enabled on model")
 
        patch_count = 0
        with torch.no_grad():
            for r in row_starts:
                for c_pos in col_starts:
                    # Extract context patch: (T_in, C, p, p)
                    ctx_patch = context[:, :, r:r+p, c_pos:c_pos+p].copy()
 
                    # Skip mostly NaN patches
                    nan_pct = np.mean(~np.isfinite(ctx_patch))
                    if nan_pct > self.config["patches"]["nan_threshold"]:
                        continue
 
                    # Fill NaN with 0 for model input
                    ctx_patch = np.where(np.isfinite(ctx_patch), ctx_patch, 0.0)
 
                    # Add validity channel if model expects it
                    if self.use_validity:
                        valid_ch = (np.isfinite(ctx_patch[:, :1])).astype(np.float32)
                        ctx_patch = np.concatenate([ctx_patch, valid_ch], axis=1)
 
                    # Gather MC samples
                    sample_preds = []   # list of (n_forecast, C, p, p)
                    for _ in range(N):
                        pred_patch = self._rollout(
                            ctx_patch, n_forecast, clim_floors, r, c_pos, p
                        )  # (n_forecast, C, p, p)
                        sample_preds.append(pred_patch)
 
                    # Stack and compute mean across MC samples
                    sample_stack = np.stack(sample_preds, axis=0)  # (N, n_forecast, C, p, p)
                    patch_mean = sample_stack.mean(axis=0)          # (n_forecast, C, p, p)
                    patch_sq   = (sample_stack ** 2).mean(axis=0)   # (n_forecast, C, p, p)
 
                    # Accumulate with Gaussian weight
                    w = weight_patch[np.newaxis, np.newaxis]  # (1, 1, p, p)
                    pred_sum   [:, :, r:r+p, c_pos:c_pos+p] += patch_mean * w
                    pred_sq_sum[:, :, r:r+p, c_pos:c_pos+p] += patch_sq   * w
                    weight_sum [:, :, r:r+p, c_pos:c_pos+p] += w
 
                    patch_count += 1
 
        # Normalize by weights
        safe_weight = np.maximum(weight_sum, 1e-10)
        cf_mean = (pred_sum    / safe_weight).astype(np.float32)
        cf_sq   = (pred_sq_sum / safe_weight).astype(np.float32)
 
        # Variance = E[X²] - E[X]²  (weighted; approximation for N samples)
        cf_var = np.maximum(cf_sq - cf_mean ** 2, 0.0)
        cf_std = np.sqrt(cf_var).astype(np.float32)
 
        # Mark zero-weight regions as NaN
        cf_mean = np.where(weight_sum < 1e-10, np.nan, cf_mean)
        cf_std  = np.where(weight_sum < 1e-10, np.nan, cf_std)
 
        logger.info(f"  Stitched {patch_count} patches successfully")
        return cf_mean, cf_std
 
    def _rollout(
        self,
        context: np.ndarray,
        n_steps: int,
        clim_floors,
        row: int,
        col: int,
        p: int,
    ) -> np.ndarray:
        """
        Autoregressive rollout: repeatedly feed model output back as input.
        Applies seasonal climatology floor after each step if enabled.
 
        context     : (T_in, C_in, p, p) — single patch context
        clim_floors : list of (C, H, W) global floors or None
        row, col    : patch position in global map (for extracting floor slice)
 
        Returns : (n_steps, C_out, p, p)
        """
        T_in, C_in, _p, _ = context.shape
        C_out = self.n_bands
 
        # Convert to tensor
        x = torch.from_numpy(context).float().unsqueeze(0).to(self.device)
        # x: (1, T_in, C_in, p, p)
 
        predictions = []
        current_context = x
 
        t_output = self.config["model"]["t_output"]
        steps_generated = 0
 
        while steps_generated < n_steps:
            # Generate next t_output steps
            pred = self.model(current_context)   # (1, T_out, C_out, p, p)
            batch_pred = pred[0].cpu().numpy()   # (T_out, C_out, p, p)
 
            steps_left = n_steps - steps_generated
            take = min(t_output, steps_left)
            take_pred = batch_pred[:take]
 
            # Apply climatology floor (step-by-step)
            if self.use_climatology and clim_floors is not None:
                for i in range(take):
                    step_idx = steps_generated + i
                    if step_idx < len(clim_floors):
                        floor_patch = clim_floors[step_idx][:, row:row+p, col:col+p]
                        take_pred[i] = np.maximum(take_pred[i], floor_patch)
 
            predictions.append(take_pred)
            steps_generated += take
 
            if steps_generated < n_steps:
                # Update context window: append predictions, drop oldest
                # If model has validity channel, add dummy valid channel to pred
                if C_in > C_out:
                    # Add validity channel (assume valid for predictions)
                    valid_ch = torch.ones(
                        1, t_output, 1, p, p, device=self.device
                    )
                    pred_with_validity = torch.cat([pred, valid_ch], dim=2)
                else:
                    pred_with_validity = pred
 
                # Roll context: drop first t_output, append new predictions
                take_t = min(t_output, current_context.shape[1])
                current_context = torch.cat(
                    [current_context[:, take_t:], pred_with_validity[:, :take_t]],
                    dim=1
                )
 
        return np.concatenate(predictions, axis=0)[:n_steps]  # (n_steps, C_out, p, p)
 
    @staticmethod
    def _gaussian_patch_weight(patch_size: int, sigma_fraction: float = 0.3) -> np.ndarray:
        """
        Generate a 2D Gaussian weighting kernel for smooth patch blending.
        Values peak at center, taper toward edges.
        """
        sigma = patch_size * sigma_fraction
        center = patch_size / 2.0
        y, x = np.mgrid[:patch_size, :patch_size]
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        weight = np.exp(-dist_sq / (2 * sigma ** 2))
        return weight.astype(np.float32)