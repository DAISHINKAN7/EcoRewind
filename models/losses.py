"""
losses.py
---------
ECO-REWIND loss function:
 
  L_total = L_reconstruction + λ₁·L_temporal_smooth + λ₂·L_ecological
          + λ₃·L_ssim + λ₄·L_spectral
 
  L_reconstruction  : weighted MSE between predicted and actual (SAR=0.1)
  L_temporal_smooth : penalises unrealistic quarter-to-quarter NDVI/NDWI jumps
  L_ecological      : soft constraints for physically valid value ranges
  L_ssim            : structural similarity — rewards spatial texture realism
  L_spectral        : enforces NDVI consistency with NIR and Red predictions
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
 
 
# ---------------------------------------------------------------------------
# SSIM helper
# ---------------------------------------------------------------------------
 
class _SSIMModule(nn.Module):
    """
    Differentiable SSIM loss computed with a fixed 11×11 Gaussian kernel.
    Applied per-band independently; averaged across bands.
 
    Returns 1 − mean_SSIM so that minimising it improves structural quality.
    """
 
    def __init__(self, window_size: int = 11, n_bands: int = 7):
        super().__init__()
        self.window_size = window_size
        self.n_bands = n_bands
 
        # Build 2D Gaussian kernel
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)   # (1, 1, ws, ws)
        self.register_buffer("window", window)
 
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : (B, T, C, H, W)
        Returns scalar SSIM loss in [0, 2] (0 = perfect match).
        """
        B, T, C, H, W = pred.shape
        # Fold B and T into batch so we process all frames at once
        p = pred.reshape(B * T, C, H, W)
        t = target.reshape(B * T, C, H, W)
 
        pad = self.window_size // 2
        ssim_vals = []
 
        for c in range(min(C, self.n_bands)):
            pc = p[:, c:c+1]   # (BT, 1, H, W)
            tc = t[:, c:c+1]
 
            mu_p  = F.conv2d(pc, self.window, padding=pad)
            mu_t  = F.conv2d(tc, self.window, padding=pad)
            mu_p2 = mu_p ** 2
            mu_t2 = mu_t ** 2
            mu_pt = mu_p * mu_t
 
            sig_p2 = F.conv2d(pc * pc, self.window, padding=pad) - mu_p2
            sig_t2 = F.conv2d(tc * tc, self.window, padding=pad) - mu_t2
            sig_pt = F.conv2d(pc * tc, self.window, padding=pad) - mu_pt
 
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            num  = (2 * mu_pt + C1) * (2 * sig_pt + C2)
            den  = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2).clamp(min=1e-8)
            ssim_map = num / den
            ssim_vals.append(ssim_map.mean())
 
        return 1.0 - torch.stack(ssim_vals).mean()   # scalar, lower = better
 
 
# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------
 
class EcoRewindLoss(nn.Module):
    """
    Combined loss with five components:
 
    1. Reconstruction : weighted MSE (SAR_VV weight=0.1), validity-masked
    2. Temporal smooth: penalises |Δt_pred − Δt_actual| for NDVI/NDWI
    3. Ecological     : soft ReLU penalty for out-of-bounds predictions
    4. SSIM           : structural similarity — sharper spatial counterfactuals
    5. Spectral       : NDVI prediction must be consistent with NIR and Red
    """
 
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.lambda_temporal   = config["loss"]["lambda_temporal"]
        self.lambda_ecological = config["loss"]["lambda_ecological"]
        self.lambda_ssim       = config["loss"].get("lambda_ssim", 0.15)
        self.lambda_spectral   = config["loss"].get("lambda_spectral", 0.05)
 
        # Band indices
        band_idx = config["bands"]["indices"]
        self.ndvi_idx = band_idx["ndvi"]
        self.ndwi_idx = band_idx["ndwi"]
        self.sar_idx  = band_idx["sar_vv"]
        self.nir_idx  = band_idx["nir"]
        self.red_idx  = band_idx["red"]
        self.n_bands  = config["bands"]["count"]
 
        self.band_bounds = self._build_bounds(config)
 
        # Per-band reconstruction weights.
        # SAR_VV gets 0.1 — high speckle noise makes pixel-wise MSE unreliable.
        weights = torch.ones(self.n_bands)
        weights[self.sar_idx] = 0.1
        self.register_buffer("band_weights", weights)
 
        # SSIM module (fixed kernel, no learnable params)
        self.ssim_module = _SSIMModule(window_size=11, n_bands=self.n_bands)
 
    def _build_bounds(self, config: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        norm_methods = config["bands"]["normalization"]
        band_names   = config["bands"]["names"]
        bounds = {}
        for name in band_names:
            method = norm_methods.get(name.lower(), "minmax")
            if method in ("minmax", "shift"):
                bounds[name.lower()] = (0.0, 1.0)
            elif method == "zscore":
                bounds[name.lower()] = (-4.0, 4.0)
        return bounds
 
    def forward(
        self,
        pred:     torch.Tensor,
        target:   torch.Tensor,
        validity: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred     : (B, T_out, C, H, W)
            target   : (B, T_out, C, H, W)
            validity : (B, T_out, 1, H, W) optional validity mask
 
        Returns:
            total_loss, components dict
        """
        if validity is not None:
            mask = validity.expand_as(pred).float()
        else:
            mask = torch.ones_like(pred)
 
        n_valid = mask.sum().clamp(min=1.0)
 
        # 1. Reconstruction loss (masked MSE, per-band weighted)
        w = self.band_weights.view(1, 1, -1, 1, 1)
        diff_sq = ((pred - target) ** 2) * mask * w
        l_recon = diff_sq.sum() / n_valid
 
        # 2. Temporal smoothness loss
        l_temporal = self._temporal_smooth_loss(pred, target, mask)
 
        # 3. Ecological constraints
        l_eco = self._ecological_loss(pred, mask)
 
        # 4. SSIM structural loss
        # Compute on raw pred/target — do NOT multiply by mask before passing to SSIM.
        # Zeroing masked pixels with pred*mask creates large uniform regions that push
        # the 11×11 kernel's SSIM above 1 (numerically impossible), giving negative loss.
        l_ssim = self.ssim_module(pred, target)
 
        # 5. Spectral consistency (disabled by default via lambda_spectral=0)
        # The formula requires physical-space NIR/Red values but the model works in
        # normalised space where NIR_max ≠ Red_max, so the consistency check
        # is not directly comparable. Only enable if you pass inverse-normalised tensors.
        l_spectral = torch.tensor(0.0, device=pred.device)
 
        # 5. Spectral consistency: NDVI_pred should match (NIR_pred - Red_pred) / (NIR_pred + Red_pred)
        l_spectral = self._spectral_consistency_loss(pred, mask)
 
        total = (l_recon
                 + self.lambda_temporal   * l_temporal
                 + self.lambda_ecological * l_eco
                 + self.lambda_ssim       * l_ssim
                 + self.lambda_spectral   * l_spectral)
 
        return total, {
            "reconstruction":   l_recon.detach(),
            "temporal_smooth":  l_temporal.detach(),
            "ecological":       l_eco.detach(),
            "ssim":             l_ssim.detach(),
            "spectral":         l_spectral.detach(),
        }
 
    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------
 
    def _temporal_smooth_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   torch.Tensor,
    ) -> torch.Tensor:
        """MSE(Δt_pred, Δt_actual) on NDVI and NDWI only."""
        if pred.shape[1] < 2:
            return torch.tensor(0.0, device=pred.device)
 
        ndvi, ndwi = self.ndvi_idx, self.ndwi_idx
 
        pred_delta_ndvi = pred[:, 1:, ndvi]   - pred[:, :-1, ndvi]
        tgt_delta_ndvi  = target[:, 1:, ndvi] - target[:, :-1, ndvi]
        pred_delta_ndwi = pred[:, 1:, ndwi]   - pred[:, :-1, ndwi]
        tgt_delta_ndwi  = target[:, 1:, ndwi] - target[:, :-1, ndwi]
 
        mask_t = mask[:, 1:, ndvi]
        n_valid = mask_t.sum().clamp(min=1.0)
 
        l  = (((pred_delta_ndvi - tgt_delta_ndvi) ** 2) * mask_t).sum() / n_valid
        l += (((pred_delta_ndwi - tgt_delta_ndwi) ** 2) * mask_t).sum() / n_valid
        return l / 2.0
 
    def _ecological_loss(
        self, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Soft ReLU penalty for predictions outside physical bounds."""
        band_names = ["blue", "green", "red", "nir", "ndvi", "ndwi", "sar_vv"]
        l_eco   = torch.tensor(0.0, device=pred.device)
        n_valid = mask[:, :, 0].sum().clamp(min=1.0)
 
        for c, name in enumerate(band_names):
            if c >= pred.shape[2]:
                break
            lo, hi = self.band_bounds.get(name, (-5.0, 5.0))
            band = pred[:, :, c]
            m    = mask[:, :, c]
            l_eco = l_eco + (
                (F.relu(lo - band) * m).sum() +
                (F.relu(band - hi) * m).sum()
            ) / n_valid
 
        return l_eco / len(band_names)
 
    def _spectral_consistency_loss(
        self, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalise predictions where independently-predicted NDVI disagrees
        with (NIR_pred − Red_pred) / (NIR_pred + Red_pred).
 
        All bands are in normalised [0,1] space (minmax/shift).
        NDVI is shift-normalised: physical = norm*2 − 1.
        NIR and Red are minmax-normalised: physical ≈ [0, ~0.6].
        """
        nir_n  = pred[:, :, self.nir_idx]    # normalised NIR  [0,1]
        red_n  = pred[:, :, self.red_idx]    # normalised Red  [0,1]
        ndvi_n = pred[:, :, self.ndvi_idx]   # normalised NDVI [0,1]
 
        # Convert normalised NDVI back to physical [-1,+1]
        ndvi_phys = ndvi_n * 2.0 - 1.0
 
        # NIR and Red are minmax — they are proportional to physical reflectance
        # so the ratio is valid even without full inverse-transform
        eps = 1e-6
        ndvi_expected = (nir_n - red_n) / (nir_n + red_n + eps)
        ndvi_expected = ndvi_expected.clamp(-1.0, 1.0)
 
        m = mask[:, :, self.ndvi_idx]
        n_valid = m.sum().clamp(min=1.0)
        return ((ndvi_phys - ndvi_expected) ** 2 * m).sum() / n_valid
 
 
def build_loss(config: Dict[str, Any]) -> EcoRewindLoss:
    return EcoRewindLoss(config)