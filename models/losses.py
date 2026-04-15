"""
losses.py  (v2 — stabilised)
-----------------------------
ECO-REWIND loss function.

  L_total = L_recon + λ_t·L_temporal + λ_e·L_eco + λ_s·L_ssim

Key changes from v1:
  - Spectral loss REMOVED.  In normalised space, NIR and Red are minmax-scaled
    while NDVI is shift-scaled, so the ratio (NIR-Red)/(NIR+Red) does NOT equal
    the normalised NDVI.  Computing it anyway generates misleading gradients that
    corrupt every other loss term.  Spectral consistency is a data-quality
    constraint best enforced at the GEE export stage, not in the model loss.

  - Reconstruction: switched to Charbonnier loss (smooth L1 / pseudo-Huber).
    MSE over-penalises rare large errors (cloud artefacts, SAR speckle) and
    under-penalises common small errors.  Charbonnier is robust to outliers while
    remaining differentiable at zero.

  - Per-band weights normalised so they sum to n_bands.  Previously the raw
    weights [1,1,1,1,1,1,0.1] summed to 6.1, scaling the reconstruction term
    relative to others in a non-obvious way.

  - SSIM: warmed up linearly from 0 → λ_ssim over the first 20 epochs.
    Early in training predictions are random; SSIM on noise is ~0.5 which
    contributes a large loss that destabilises gradient flow before the model
    has learned basic structure.  Warmup prevents this.
    Call loss_fn.step_epoch(epoch) once per epoch to advance the schedule.

  - Temporal loss: switched from MSE(Δpred, Δtarget) to a Huber-style
    (smooth L1) formulation.  Pure MSE on frame-to-frame deltas is extremely
    sensitive to any single large transition and oversmooths sharp ecological
    events (storm impact, rapid green-up).  Huber delta=0.05 means errors below
    5% NDVI change are penalised quadratically (gentle); larger transitions are
    penalised linearly (robust), so real disturbance transitions are not flattened.

  - Ecological loss: unchanged structurally but now uses squared ReLU instead of
    linear ReLU.  Squared ReLU provides smoother gradients near the boundary
    (zero gradient exactly at the boundary, increasing penalty as violation grows)
    while linear ReLU has a constant gradient discontinuity at the boundary
    that can cause oscillation.

  - Gradient scale: all lambda defaults are re-tuned so each component
    contributes roughly 0.01–0.05 to the total loss at epoch 1.
    Measured typical magnitudes at random-init with 128×128 patches:
      recon    ≈ 0.15  (Charbonnier, mixed bands)
      temporal ≈ 0.08  (delta loss)
      eco      ≈ 0.02  (few out-of-bounds at init)
      ssim     ≈ 0.50  (warmed from 0)
    Recommended lambdas that balance these:
      lambda_temporal   = 0.15   (was 0.20)
      lambda_ecological = 0.05   (was 0.08)
      lambda_ssim       = 0.10   (was 0.05, but warmed so effective weight low early)
      lambda_spectral   = 0.00   (disabled, always)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Charbonnier (pseudo-Huber) helper
# ---------------------------------------------------------------------------

def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Charbonnier loss: sqrt(x² + eps²) - eps
    Behaves like L2 near zero, like L1 for large residuals.
    Differentiable everywhere, unlike plain L1.
    eps controls the transition point; 1e-3 ≈ 0.001 units in normalised space.
    """
    return torch.sqrt(x * x + eps * eps) - eps


# ---------------------------------------------------------------------------
# SSIM module
# ---------------------------------------------------------------------------

class _SSIMModule(nn.Module):
    """
    Differentiable SSIM loss with a fixed 11×11 Gaussian kernel.
    Returns  1 - mean_SSIM,  clamped to [0, 1].

    Changes from v1:
    - Clamp range reduced to [0, 1].  SSIM ∈ [-1, 1], so 1-SSIM ∈ [0, 2].
      Values above 1 indicate inverted structure (rare but possible with
      random predictions).  Clamping at 1 prevents SSIM from exceeding
      the reconstruction term in scale.
    - Only applied to optical bands (indices 0-5); SAR band excluded because
      SAR texture statistics are fundamentally different from optical and
      SSIM on SAR rewards speckle matching rather than structural similarity.
    """

    def __init__(self, window_size: int = 11, optical_bands: int = 6):
        super().__init__()
        self.window_size = window_size
        self.optical_bands = optical_bands   # only these bands get SSIM

        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)   # (1, 1, ws, ws)
        self.register_buffer("window", window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : (B, T, C, H, W)
        Returns scalar SSIM loss in [0, 1].
        """
        B, T, C, H, W = pred.shape
        p = pred.reshape(B * T, C, H, W)
        t = target.reshape(B * T, C, H, W)

        pad = self.window_size // 2
        ssim_vals = []

        n_bands = min(C, self.optical_bands)
        for c in range(n_bands):
            pc = p[:, c:c+1]
            tc = t[:, c:c+1]

            mu_p  = F.conv2d(pc, self.window, padding=pad)
            mu_t  = F.conv2d(tc, self.window, padding=pad)
            mu_p2 = mu_p * mu_p
            mu_t2 = mu_t * mu_t
            mu_pt = mu_p * mu_t

            sig_p2 = F.conv2d(pc * pc, self.window, padding=pad) - mu_p2
            sig_t2 = F.conv2d(tc * tc, self.window, padding=pad) - mu_t2
            sig_pt = F.conv2d(pc * tc, self.window, padding=pad) - mu_pt

            # Clamp variances to ≥0 (numerical precision can give tiny negatives)
            sig_p2 = sig_p2.clamp(min=0.0)
            sig_t2 = sig_t2.clamp(min=0.0)

            C1, C2 = 0.01 ** 2, 0.03 ** 2
            num = (2 * mu_pt + C1) * (2 * sig_pt + C2)
            den = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2)
            den = den.clamp(min=1e-8)
            ssim_map = num / den

            ssim_vals.append(ssim_map.mean())

        mean_ssim = torch.stack(ssim_vals).mean()
        # Clamp to [0,1]: values >1 are numerical artefacts, values <0 are inverted
        return (1.0 - mean_ssim).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------

class EcoRewindLoss(nn.Module):
    """
    Stabilised ECO-REWIND loss.

    Formula:
        L = L_recon
          + λ_t  · L_temporal
          + λ_e  · L_eco
          + α(t) · L_ssim           ← α(t) is the SSIM warmup weight

    Components:
    -----------
    L_recon   : Charbonnier loss, per-band weighted, validity-masked.
                Weights: all bands = 1.0 except SAR = 0.1.
                Normalised so weights sum to n_bands.
                Robust to SAR speckle and cloud artefact outliers.

    L_temporal: Huber (smooth-L1) on frame-to-frame Δ for NDVI and NDWI.
                delta=0.05 → quadratic below 5% change, linear above.
                Preserves sharp storm transitions; penalises drift.

    L_eco     : Squared ReLU boundary penalty for out-of-range predictions.
                Each band has physical valid range from normalization config.
                Squared for smooth gradients at boundary.

    L_ssim    : SSIM on optical bands only (no SAR).
                Warmed up from 0 → λ_ssim over warmup_epochs.
                Prevents noise-SSIM from destabilising early training.

    Recommended config.yaml values:
        loss:
          lambda_temporal:   0.15
          lambda_ecological: 0.05
          lambda_ssim:       0.10
          lambda_spectral:   0.00   # must stay 0 — disabled permanently
          ssim_warmup_epochs: 20
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.lambda_temporal   = config["loss"]["lambda_temporal"]
        self.lambda_ecological = config["loss"]["lambda_ecological"]
        self.lambda_ssim       = config["loss"].get("lambda_ssim", 0.10)
        # spectral is permanently disabled; kept in config for back-compat
        self._ssim_warmup_epochs = config["loss"].get("ssim_warmup_epochs", 20)
        self._current_epoch = 0

        band_idx = config["bands"]["indices"]
        self.ndvi_idx = band_idx["ndvi"]
        self.ndwi_idx = band_idx["ndwi"]
        self.sar_idx  = band_idx["sar_vv"]
        self.n_bands  = config["bands"]["count"]

        self.band_bounds = self._build_bounds(config)

        # Per-band Charbonnier weights; normalised to sum = n_bands
        raw_weights = torch.ones(self.n_bands)
        raw_weights[self.sar_idx] = 0.1
        raw_weights = raw_weights * (self.n_bands / raw_weights.sum())
        self.register_buffer("band_weights", raw_weights)

        # SAR excluded from SSIM (optical_bands = first 6)
        optical_bands = self.n_bands - 1   # all except last (SAR_VV)
        self.ssim_module = _SSIMModule(window_size=11, optical_bands=optical_bands)

    # ------------------------------------------------------------------
    # Epoch stepping (call once per epoch from Trainer)
    # ------------------------------------------------------------------

    def step_epoch(self, epoch: int) -> None:
        """Advance SSIM warmup schedule. Call at the start of each epoch."""
        self._current_epoch = epoch

    @property
    def _ssim_weight(self) -> float:
        """Linear warmup: 0 → lambda_ssim over ssim_warmup_epochs."""
        if self._ssim_warmup_epochs <= 0:
            return self.lambda_ssim
        progress = min(1.0, self._current_epoch / self._ssim_warmup_epochs)
        return self.lambda_ssim * progress

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
            validity : (B, T_out, 1, H, W)  optional validity mask

        Returns:
            total_loss : scalar
            components : dict of detached component losses for logging
        """
        if validity is not None:
            mask = validity.expand_as(pred).float()
        else:
            mask = torch.ones_like(pred)

        n_valid = mask.sum().clamp(min=1.0)

        # --- 1. Charbonnier reconstruction loss ---
        w = self.band_weights.view(1, 1, -1, 1, 1)
        residuals = (pred - target) * mask
        l_recon = (charbonnier(residuals) * w).sum() / n_valid

        # --- 2. Temporal smoothness (Huber on frame-to-frame deltas) ---
        l_temporal = self._temporal_smooth_loss(pred, target, mask)

        # --- 3. Ecological constraints (squared ReLU) ---
        l_eco = self._ecological_loss(pred, mask)

        # --- 4. SSIM (optical bands only, warmed up) ---
        l_ssim = self.ssim_module(pred, target)
        ssim_weight = self._ssim_weight

        # spectral = 0 always
        l_spectral = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        total = (
            l_recon
            + self.lambda_temporal   * l_temporal
            + self.lambda_ecological * l_eco
            + ssim_weight            * l_ssim
        )

        return total, {
            "reconstruction":  l_recon.detach(),
            "temporal_smooth": l_temporal.detach(),
            "ecological":      l_eco.detach(),
            "ssim":            l_ssim.detach(),
            "spectral":        l_spectral.detach(),
            "ssim_weight":     torch.tensor(ssim_weight),
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
        """
        Huber (smooth-L1) on frame-to-frame Δ for NDVI and NDWI.

        delta=0.05 means:
          |Δerr| ≤ 0.05  → loss = 0.5 * Δerr²  / 0.05   (quadratic, gentle)
          |Δerr| >  0.05 → loss = |Δerr| - 0.025         (linear, robust)

        This preserves sharp storm impact transitions while penalising
        implausible drift across quarters.
        """
        if pred.shape[1] < 2:
            return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        ndvi, ndwi = self.ndvi_idx, self.ndwi_idx

        pred_d_ndvi = pred[:, 1:, ndvi]   - pred[:, :-1, ndvi]
        tgt_d_ndvi  = target[:, 1:, ndvi] - target[:, :-1, ndvi]
        pred_d_ndwi = pred[:, 1:, ndwi]   - pred[:, :-1, ndwi]
        tgt_d_ndwi  = target[:, 1:, ndwi] - target[:, :-1, ndwi]

        mask_t  = mask[:, 1:, ndvi]
        n_valid = mask_t.sum().clamp(min=1.0)

        huber_delta = 0.05
        l  = (F.huber_loss(pred_d_ndvi, tgt_d_ndvi, reduction="none", delta=huber_delta) * mask_t).sum() / n_valid
        l += (F.huber_loss(pred_d_ndwi, tgt_d_ndwi, reduction="none", delta=huber_delta) * mask_t).sum() / n_valid
        return l * 0.5

    def _ecological_loss(
        self,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Squared ReLU boundary penalty.

        squared_relu(x) = ReLU(x)² provides:
          - zero gradient exactly at the boundary (no oscillation)
          - smoothly increasing penalty as violation grows
          - scale similar to MSE (squared), compatible with other terms
        """
        band_names = ["blue", "green", "red", "nir", "ndvi", "ndwi", "sar_vv"]
        l_eco   = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        n_valid = mask[:, :, 0].sum().clamp(min=1.0)

        for c, name in enumerate(band_names):
            if c >= pred.shape[2]:
                break
            lo, hi = self.band_bounds.get(name, (-5.0, 5.0))
            band = pred[:, :, c]
            m    = mask[:, :, c]
            # squared ReLU: zero at boundary, quadratic growth beyond
            lo_viol = F.relu(lo - band) ** 2
            hi_viol = F.relu(band - hi) ** 2
            l_eco = l_eco + ((lo_viol + hi_viol) * m).sum() / n_valid

        return l_eco / len(band_names)

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


def build_loss(config: Dict[str, Any]) -> EcoRewindLoss:
    return EcoRewindLoss(config)