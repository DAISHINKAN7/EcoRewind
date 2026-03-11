"""
losses.py
---------
ECO-REWIND loss function:
 
  L_total = L_reconstruction + λ₁·L_temporal_smooth + λ₂·L_ecological
 
  L_reconstruction  : MSE between predicted and actual (pre-disturbance)
  L_temporal_smooth : penalises unrealistic quarter-to-quarter jumps
  L_ecological      : soft constraints for physically valid value ranges
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
 
 
class EcoRewindLoss(nn.Module):
    """
    Combined loss with three components:
 
    1. Reconstruction: MSE on all bands, masked by validity
    2. Temporal smooth: penalises |Δt_pred - Δt_actual| for NDVI/NDWI
    3. Ecological:
         - Band range violations (bands outside physical limits)
         - Biomass monotonicity (soft: NDVI shouldn't plummet unrealistically)
    """
 
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.lambda_temporal = config["loss"]["lambda_temporal"]
        self.lambda_ecological = config["loss"]["lambda_ecological"]
 
        # Band indices (post-normalization — normalized values)
        band_idx = config["bands"]["indices"]
        self.ndvi_idx = band_idx["ndvi"]
        self.ndwi_idx = band_idx["ndwi"]
        self.sar_idx = band_idx["sar_vv"]
        self.n_bands = config["bands"]["count"]
 
        # After normalization: reflectance bands → [0,1], shift bands → [0,1]
        # Physical bounds in normalized space:
        #   reflectance (minmax): [0, 1]
        #   NDVI/NDWI (shift):    [0, 1]  (−1→0, +1→1)
        #   SAR_VV (zscore):      roughly [−3, 3] (3-sigma bounds)
        self.band_bounds = self._build_bounds(config)
 
    def _build_bounds(self, config: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Physical bounds in NORMALIZED space.
        After normalization, all bands should be in known ranges.
        """
        norm_methods = config["bands"]["normalization"]
        band_names = config["bands"]["names"]
        bounds = {}
        for name in band_names:
            method = norm_methods.get(name.lower(), "minmax")
            if method in ("minmax", "shift"):
                bounds[name.lower()] = (0.0, 1.0)
            elif method == "zscore":
                bounds[name.lower()] = (-4.0, 4.0)   # 4-sigma soft limit
        return bounds
 
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        validity: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred     : (B, T_out, C, H, W) — model predictions
            target   : (B, T_out, C, H, W) — ground truth
            validity : (B, T_out, 1, H, W) — optional validity mask
 
        Returns:
            total_loss : scalar
            components : dict {"reconstruction", "temporal_smooth", "ecological"}
        """
        # Build validity mask for loss weighting
        if validity is not None:
            # validity: (B, T, 1, H, W) — expand to match pred
            mask = validity.expand_as(pred).float()
        else:
            mask = torch.ones_like(pred)
 
        n_valid = mask.sum().clamp(min=1.0)
 
        # --- 1. Reconstruction loss (masked MSE) ---
        diff_sq = ((pred - target) ** 2) * mask
        l_recon = diff_sq.sum() / n_valid
 
        # --- 2. Temporal smoothness loss ---
        # Penalise unrealistic frame-to-frame changes in NDVI and NDWI
        l_temporal = self._temporal_smooth_loss(pred, target, mask)
 
        # --- 3. Ecological constraints ---
        l_eco = self._ecological_loss(pred, mask)
 
        # --- Combine ---
        total = l_recon + self.lambda_temporal * l_temporal + self.lambda_ecological * l_eco
 
        return total, {
            "reconstruction": l_recon.detach(),
            "temporal_smooth": l_temporal.detach(),
            "ecological": l_eco.detach(),
        }
 
    def _temporal_smooth_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_temporal = MSE(Δt_pred, Δt_actual)
        Only on NDVI and NDWI (ecologically meaningful temporal trends).
        """
        if pred.shape[1] < 2:
            return torch.tensor(0.0, device=pred.device)
 
        ndvi = self.ndvi_idx
        ndwi = self.ndwi_idx
 
        pred_delta_ndvi = pred[:, 1:, ndvi] - pred[:, :-1, ndvi]   # (B, T-1, H, W)
        tgt_delta_ndvi = target[:, 1:, ndvi] - target[:, :-1, ndvi]
 
        pred_delta_ndwi = pred[:, 1:, ndwi] - pred[:, :-1, ndwi]
        tgt_delta_ndwi = target[:, 1:, ndwi] - target[:, :-1, ndwi]
 
        mask_t = mask[:, 1:, ndvi]  # (B, T-1, H, W)
        n_valid = mask_t.sum().clamp(min=1.0)
 
        l = (((pred_delta_ndvi - tgt_delta_ndvi) ** 2) * mask_t).sum() / n_valid
        l += (((pred_delta_ndwi - tgt_delta_ndwi) ** 2) * mask_t).sum() / n_valid
        return l / 2.0
 
    def _ecological_loss(
        self, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft penalty for predictions outside physical bounds.
 
        Uses ReLU to only penalise actual violations (not near-boundary values).
        """
        band_names = [
            "blue", "green", "red", "nir", "ndvi", "ndwi", "sar_vv"
        ]
        l_eco = torch.tensor(0.0, device=pred.device)
        n_valid = mask[:, :, 0].sum().clamp(min=1.0)
 
        for c, name in enumerate(band_names):
            if c >= pred.shape[2]:
                break
            lo, hi = self.band_bounds.get(name, (-5.0, 5.0))
            band = pred[:, :, c]         # (B, T, H, W)
            m = mask[:, :, c]            # (B, T, H, W)
 
            # Penalise below lower bound
            violation_lo = F.relu(lo - band) * m
            # Penalise above upper bound
            violation_hi = F.relu(band - hi) * m
 
            l_eco = l_eco + (violation_lo.sum() + violation_hi.sum()) / n_valid
 
        return l_eco / len(band_names)
 
 
def build_loss(config: Dict[str, Any]) -> EcoRewindLoss:
    return EcoRewindLoss(config)