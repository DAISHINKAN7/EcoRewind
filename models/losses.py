"""
losses.py  (v3 — full stability overhaul)
------------------------------------------
Key changes vs v2:

FIX B — Loss normalized by EXPECTED pixel count, not masked count.
  Old: loss = sum(residuals * mask) / mask.sum()
       When mask is sparse (few valid pixels), denominator → 0 → loss → 0 or NaN.
       EarlyStop then treats val_loss=0.0 as "perfect model" → degenerate collapse.
  New: loss = sum(residuals * mask) / (B * T * H * W * optical_band_fraction)
       Denominator is FIXED (expected pixels), so sparse masks produce SMALL loss,
       not artificially zero loss. This is correct behaviour: a sparse batch
       contributes proportionally less to the gradient, not a fake zero.

FIX B2 — NDVI-specific loss weighting (3×).
  NDVI is the primary ecological signal. Up-weighting it ensures the model
  prioritises it over less informative bands (Blue, SAR fill values).
  Band weights: Blue=0.5, Green=0.8, Red=0.8, NIR=1.0, NDVI=3.0, NDWI=2.0, SAR=0.0
  (SAR weight=0 since it's excluded from loss entirely)

FIX A — Diagnostics: log % valid pixels, loss before/after masking, per-band contributions.
  step_epoch() now also accepts batch-level info for W&B logging.

FIX G — val_loss < 1e-6 guard: the loss function returns a sentinel
  DEGENERATE_LOSS flag so the trainer can detect and skip it.

UNCHANGED: Charbonnier, SSIM warmup, temporal Huber, ecological squared-ReLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Sentinel value returned when a batch is degenerate
DEGENERATE_LOSS_SENTINEL = -1.0


# ---------------------------------------------------------------------------
# Charbonnier helper
# ---------------------------------------------------------------------------

def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps) - eps


# ---------------------------------------------------------------------------
# SSIM module (unchanged from v2)
# ---------------------------------------------------------------------------

class _SSIMModule(nn.Module):
    def __init__(self, window_size: int = 11, optical_bands: int = 6):
        super().__init__()
        self.window_size = window_size
        self.optical_bands = optical_bands
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
            mu_p2, mu_t2, mu_pt = mu_p*mu_p, mu_t*mu_t, mu_p*mu_t
            sig_p2 = (F.conv2d(pc*pc, self.window, padding=pad) - mu_p2).clamp(min=0)
            sig_t2 = (F.conv2d(tc*tc, self.window, padding=pad) - mu_t2).clamp(min=0)
            sig_pt =  F.conv2d(pc*tc, self.window, padding=pad) - mu_pt
            C1, C2 = 0.01**2, 0.03**2
            num = (2*mu_pt + C1) * (2*sig_pt + C2)
            den = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2)
            ssim_vals.append((num / den.clamp(min=1e-8)).mean())
        return (1.0 - torch.stack(ssim_vals).mean()).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------

class EcoRewindLoss(nn.Module):
    """
    ECO-REWIND loss v3.

    Critical change: loss is normalized by EXPECTED pixel count
    (B * T * H * W) rather than the number of valid (masked) pixels.
    This prevents degenerate collapse when masks are sparse.

    Band weights (sum to n_bands for scale compatibility):
      Blue=0.5, Green=0.8, Red=0.8, NIR=1.0, NDVI=3.0, NDWI=2.0, SAR=0.0
      Total raw = 8.1  → normalised to sum=7 (n_bands excluding SAR)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.lambda_temporal   = config["loss"]["lambda_temporal"]
        self.lambda_ecological = config["loss"]["lambda_ecological"]
        self.lambda_ssim       = config["loss"].get("lambda_ssim", 0.10)
        self._ssim_warmup_epochs = config["loss"].get("ssim_warmup_epochs", 20)
        self._current_epoch = 0

        band_idx = config["bands"]["indices"]
        self.ndvi_idx = band_idx["ndvi"]
        self.ndwi_idx = band_idx["ndwi"]
        self.sar_idx  = band_idx["sar_vv"]
        self.n_bands  = config["bands"]["count"]

        self.band_bounds = self._build_bounds(config)
        self.band_names  = [n.lower() for n in config["bands"]["names"]]

        # FIX B2: NDVI-specific up-weighting
        # Raw weights: Blue=0.5, Green=0.8, Red=0.8, NIR=1.0, NDVI=3.0, NDWI=2.0, SAR=0.0
        raw_w = torch.ones(self.n_bands)
        raw_w[band_idx["blue"]]  = 0.5
        raw_w[band_idx["green"]] = 0.8
        raw_w[band_idx["red"]]   = 0.8
        raw_w[band_idx["nir"]]   = 1.0
        raw_w[band_idx["ndvi"]]  = 3.0
        raw_w[band_idx["ndwi"]]  = 2.0
        raw_w[self.sar_idx]      = 0.0   # SAR excluded from loss
        # Normalise so non-SAR weights sum to n_optical_bands
        n_optical = self.n_bands - 1
        optical_sum = raw_w.sum()
        raw_w = raw_w * (n_optical / optical_sum.clamp(min=1e-6))
        self.register_buffer("band_weights", raw_w)

        optical_bands = self.n_bands - 1
        self.ssim_module = _SSIMModule(window_size=11, optical_bands=optical_bands)

        # Diagnostic accumulators (reset each epoch)
        self._valid_pct_history = []

    # ------------------------------------------------------------------
    # Epoch stepping
    # ------------------------------------------------------------------

    def step_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch
        if self._valid_pct_history:
            mean_valid = sum(self._valid_pct_history) / len(self._valid_pct_history)
            logger.info(
                f"[Loss] Epoch {epoch} avg valid pixel% = {mean_valid*100:.1f}% "
                f"over {len(self._valid_pct_history)} batches"
            )
        self._valid_pct_history = []

    @property
    def _ssim_weight(self) -> float:
        if self._ssim_warmup_epochs <= 0:
            return self.lambda_ssim
        return self.lambda_ssim * min(1.0, self._current_epoch / self._ssim_warmup_epochs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred:     torch.Tensor,
        target:   torch.Tensor,
        validity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred     : (B, T_out, C, H, W)
            target   : (B, T_out, C, H, W)
            validity : (B, T_out, 1, H, W) or (B, T_out, C, H, W)  optional

        Returns:
            total_loss : scalar  (DEGENERATE_LOSS_SENTINEL if batch is degenerate)
            components : dict of detached scalars for logging
        """
        B, T, C, H, W = pred.shape

        # --- Build mask ---
        if validity is not None:
            # Expand validity to (B, T, C, H, W)
            if validity.shape[2] == 1:
                mask = validity.expand(B, T, C, H, W).float()
            else:
                mask = validity.float()
        else:
            mask = torch.ones_like(pred)

        # Zero out SAR from mask (never train on SAR)
        if self.sar_idx < C:
            mask = mask.clone()
            mask[:, :, self.sar_idx] = 0.0

        # FIX A: Compute and log valid pixel fraction
        n_optical_pixels = B * T * (C - 1) * H * W  # exclude SAR
        n_valid = mask[:, :, :self.sar_idx].sum().item() + \
                  mask[:, :, self.sar_idx+1:].sum().item()
        valid_pct = n_valid / max(n_optical_pixels, 1)
        self._valid_pct_history.append(valid_pct)

        # FIX G: Guard against degenerate batches (< 5% valid optical pixels)
        MIN_VALID_RATIO = 0.05
        if valid_pct < MIN_VALID_RATIO:
            sentinel = torch.tensor(
                DEGENERATE_LOSS_SENTINEL, device=pred.device, dtype=pred.dtype
            )
            return sentinel, {
                "reconstruction":  torch.tensor(0.0),
                "temporal_smooth": torch.tensor(0.0),
                "ecological":      torch.tensor(0.0),
                "ssim":            torch.tensor(0.0),
                "spectral":        torch.tensor(0.0),
                "ssim_weight":     torch.tensor(self._ssim_weight),
                "valid_pct":       torch.tensor(valid_pct),
                "is_degenerate":   torch.tensor(1.0),
            }

        # FIX B: Expected pixel count (FIXED denominator, not masked count)
        # Using expected rather than actual prevents artificially zero loss
        # when the mask is sparse. The mask down-weights sparse batches
        # proportionally rather than producing loss=0.
        n_expected = float(B * T * H * W)

        # --- 1. Charbonnier reconstruction (FIXED denominator) ---
        w = self.band_weights.view(1, 1, -1, 1, 1)
        residuals = (pred - target) * mask
        # FIX A: Loss before vs after masking for diagnostics
        l_recon_unmasked = charbonnier(pred - target).mean()
        l_recon = (charbonnier(residuals) * w).sum() / n_expected

        # --- 2. Temporal smoothness ---
        l_temporal = self._temporal_smooth_loss(pred, target, mask, n_expected)

        # --- 3. Ecological constraints ---
        l_eco = self._ecological_loss(pred, mask, n_expected)

        # --- 4. SSIM ---
        l_ssim = self.ssim_module(pred, target)
        ssim_weight = self._ssim_weight

        # --- Per-band loss for diagnostics ---
        per_band = {}
        for c, name in enumerate(self.band_names):
            if c >= C or c == self.sar_idx:
                continue
            band_res = (pred[:, :, c] - target[:, :, c]) * mask[:, :, c]
            per_band[f"loss_{name}"] = (charbonnier(band_res).sum() / n_expected).detach()

        total = (
            l_recon
            + self.lambda_temporal   * l_temporal
            + self.lambda_ecological * l_eco
            + ssim_weight            * l_ssim
        )

        # FIX G: Guard against numerically degenerate total loss
        if not torch.isfinite(total) or total.item() < 1e-9:
            sentinel = torch.tensor(
                DEGENERATE_LOSS_SENTINEL, device=pred.device, dtype=pred.dtype
            )
            return sentinel, {
                "reconstruction":  l_recon.detach(),
                "temporal_smooth": l_temporal.detach(),
                "ecological":      l_eco.detach(),
                "ssim":            l_ssim.detach(),
                "spectral":        torch.tensor(0.0),
                "ssim_weight":     torch.tensor(ssim_weight),
                "valid_pct":       torch.tensor(valid_pct),
                "is_degenerate":   torch.tensor(1.0),
                **per_band,
            }

        components = {
            "reconstruction":    l_recon.detach(),
            "recon_unmasked":    l_recon_unmasked.detach(),  # FIX A diagnostic
            "temporal_smooth":   l_temporal.detach(),
            "ecological":        l_eco.detach(),
            "ssim":              l_ssim.detach(),
            "spectral":          torch.zeros(1, device=pred.device).squeeze(),
            "ssim_weight":       torch.tensor(ssim_weight),
            "valid_pct":         torch.tensor(valid_pct),
            "is_degenerate":     torch.tensor(0.0),
            **per_band,
        }

        return total, components

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------

    def _temporal_smooth_loss(
        self,
        pred:       torch.Tensor,
        target:     torch.Tensor,
        mask:       torch.Tensor,
        n_expected: float,
    ) -> torch.Tensor:
        if pred.shape[1] < 2:
            return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        ndvi, ndwi = self.ndvi_idx, self.ndwi_idx
        pred_d_ndvi = pred[:, 1:, ndvi]   - pred[:, :-1, ndvi]
        tgt_d_ndvi  = target[:, 1:, ndvi] - target[:, :-1, ndvi]
        pred_d_ndwi = pred[:, 1:, ndwi]   - pred[:, :-1, ndwi]
        tgt_d_ndwi  = target[:, 1:, ndwi] - target[:, :-1, ndwi]

        mask_t = mask[:, 1:, ndvi]
        # Use fixed denominator for consistency
        n_t = float(pred.shape[0] * (pred.shape[1]-1) * pred.shape[3] * pred.shape[4])

        huber_delta = 0.05
        l  = (F.huber_loss(pred_d_ndvi, tgt_d_ndvi, reduction="none", delta=huber_delta) * mask_t).sum() / n_t
        l += (F.huber_loss(pred_d_ndwi, tgt_d_ndwi, reduction="none", delta=huber_delta) * mask_t).sum() / n_t
        return l * 0.5

    def _ecological_loss(
        self,
        pred:       torch.Tensor,
        mask:       torch.Tensor,
        n_expected: float,
    ) -> torch.Tensor:
        l_eco = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        for c, name in enumerate(self.band_names):
            if c >= pred.shape[2]:
                break
            lo, hi = self.band_bounds.get(name, (-5.0, 5.0))
            band = pred[:, :, c]
            m    = mask[:, :, c]
            lo_viol = F.relu(lo - band) ** 2
            hi_viol = F.relu(band - hi) ** 2
            l_eco = l_eco + ((lo_viol + hi_viol) * m).sum() / n_expected
        return l_eco / max(len(self.band_names), 1)

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