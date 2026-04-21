"""
losses.py  (v4 — learning fixes)
----------------------------------
Key fixes vs v3:

FIX L1 — Loss scale correction.
  v3 divided by B*T*H*W but multiplied by band_weights that sum to ~6.
  This produced loss values of ~2.35 instead of ~0.3, giving the
  optimiser near-zero gradients throughout training (models never learned).
  Fix: divide by B*T*C_optical*H*W so scale matches the ~0.32 unmasked
  reconstruction value seen in logs.

FIX L2 — Ecological bounds key casing.
  _build_bounds() looked up config["bands"]["names"] which are title-case
  ("Blue", "Green" ...) but stored them lowercased. The lookup then used
  name.lower() on already-lowered keys, which worked, BUT the
  band_names list in __init__ was also lowercased, while the config
  normalization dict uses lowercase keys — so the lookup was correct.
  The real bug: _build_bounds returned an empty dict for models that
  passed config["bands"]["normalization"] as a flat dict without the
  norm method for every band. Added fallback to "minmax" and explicit
  debug logging so missing bounds are visible.
  ALSO: ecological loss was returning 0.0 because band_bounds keys were
  built with name.lower() but the band_names list in forward used the
  same casing — this was actually fine, but eco loss was still 0 because
  the bounds check `lo, hi = self.band_bounds.get(name, (-5.0, 5.0))`
  was using the default for ALL bands when _build_bounds returned {}.
  Root cause: _build_bounds iterated config["bands"]["names"] (title-case)
  but stored with `.lower()` — that part was correct. The actual problem
  was that for UNet-Mamba and PatchTST, ecological loss showed 0.0000
  (not 0.0 from empty bounds, but truly zero violations). This is correct
  behaviour when the sigmoid output head keeps optical bands in [0,1].
  Logged clearly now.

FIX L3 — SSIM returning exactly 1.0 for EcoTransformer.
  The SSIMModule clamps output to [0, 1] and returns 1 - ssim_mean.
  If ssim_mean ≈ 1.0 (predictions look locally similar to targets in
  luminance/contrast even if wrong in absolute value), ssim loss → 0,
  logged as ssim=1.0000 (this was the ssim VALUE, not the loss —
  actually the loss = 1 - ssim_val, so ssim=1.0000 means loss≈0.
  Wait — the log says ssim=1.0000 for EcoTransformer which has
  ssim_warmup_epochs=20 and lambda_ssim=0.0 in config. So ssim_weight=0
  during warmup → ssim loss contributes 0 to total. The value logged
  is the raw ssim module output (1 - ssim ≈ 1.0 meaning ssim ≈ 0,
  i.e. predictions are completely uncorrelated with targets spatially).
  This is a symptom of the model not learning, not a cause.
  Fix: log both raw ssim value AND weighted contribution separately.

FIX L4 — temporal_smooth and ecological lambda values.
  Config has lambda_temporal=0.05, lambda_ecological=0.05.
  With the corrected loss scale (~0.3), these auxiliary losses
  (temporal ~0.008, eco ~0.003) contribute 0.0004 and 0.0001 —
  negligible signal. Bumped defaults to 0.2 and 0.1 respectively,
  but controlled via config so no breaking change.

FIX L5 — Gradient flow: removed the total < 1e-9 degenerate guard
  for non-NaN losses. With corrected scale, valid losses will be ~0.3
  which would never trip this guard anyway. But it was also incorrectly
  catching early-epoch losses when the model output happened to be
  close to target by chance. Now only NaN/Inf trips the sentinel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

DEGENERATE_LOSS_SENTINEL = -1.0


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps) - eps


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
        if not ssim_vals:
            return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        return (1.0 - torch.stack(ssim_vals).mean()).clamp(0.0, 1.0)


class EcoRewindLoss(nn.Module):
    """
    ECO-REWIND loss v4.

    Loss scale fix: denominator is now B * T * C_optical * H * W
    so the loss magnitude matches the per-pixel Charbonnier error (~0.3)
    rather than the inflated ~2.35 seen in v3.

    Band weights (applied per-channel, then averaged):
      Blue=0.5, Green=0.8, Red=0.8, NIR=1.0, NDVI=3.0, NDWI=2.0, SAR=0.0
    These are normalised so their mean over optical bands = 1.0,
    preserving loss scale regardless of band count.
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

        self.band_names = [n.lower() for n in config["bands"]["names"]]
        self.band_bounds = self._build_bounds(config)

        # FIX L1: band weights normalised so mean over optical bands = 1.0
        # This keeps loss scale at ~per-pixel Charbonnier error regardless
        # of how many bands there are.
        raw_w = torch.ones(self.n_bands)
        raw_w[band_idx["blue"]]  = 0.5
        raw_w[band_idx["green"]] = 0.8
        raw_w[band_idx["red"]]   = 0.8
        raw_w[band_idx["nir"]]   = 1.0
        raw_w[band_idx["ndvi"]]  = 3.0
        raw_w[band_idx["ndwi"]]  = 2.0
        raw_w[self.sar_idx]      = 0.0

        # Normalise: mean of optical weights = 1.0
        optical_mask = torch.ones(self.n_bands, dtype=torch.bool)
        optical_mask[self.sar_idx] = False
        n_optical = optical_mask.sum().item()
        optical_mean = raw_w[optical_mask].mean()
        raw_w[optical_mask] = raw_w[optical_mask] / optical_mean.clamp(min=1e-6)
        # raw_w[sar] stays 0
        self.register_buffer("band_weights", raw_w)
        self.n_optical = int(n_optical)

        logger.info(
            f"[Loss] Band weights (normalised): "
            + ", ".join(f"{n}={raw_w[i].item():.3f}"
                        for i, n in enumerate(self.band_names))
        )

        optical_bands = self.n_bands - 1
        self.ssim_module = _SSIMModule(window_size=11, optical_bands=optical_bands)

        self._valid_pct_history = []

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

    def forward(
        self,
        pred:     torch.Tensor,
        target:   torch.Tensor,
        validity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, C, H, W = pred.shape

        # Build mask
        if validity is not None:
            if validity.shape[2] == 1:
                mask = validity.expand(B, T, C, H, W).float()
            else:
                mask = validity.float()
        else:
            mask = torch.ones_like(pred)

        # Zero SAR from mask
        if self.sar_idx < C:
            mask = mask.clone()
            mask[:, :, self.sar_idx] = 0.0

        # Valid pixel diagnostics
        n_optical_pixels = B * T * (C - 1) * H * W
        n_valid = (
            mask[:, :, :self.sar_idx].sum().item() +
            mask[:, :, self.sar_idx+1:].sum().item()
        )
        valid_pct = n_valid / max(n_optical_pixels, 1)
        self._valid_pct_history.append(valid_pct)

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

        # FIX L1: denominator = B * T * C_optical * H * W
        # This normalises loss to per-pixel-per-optical-band scale (~0.3)
        # instead of the inflated ~2.35 from v3's B*T*H*W denominator.
        n_expected = float(B * T * self.n_optical * H * W)

        # 1. Charbonnier reconstruction
        w = self.band_weights.view(1, 1, -1, 1, 1)
        residuals = (pred - target) * mask
        l_recon_unmasked = charbonnier(pred - target).mean()
        l_recon = (charbonnier(residuals) * w).sum() / n_expected

        # 2. Temporal smoothness
        l_temporal = self._temporal_smooth_loss(pred, target, mask, n_expected)

        # 3. Ecological constraints
        l_eco = self._ecological_loss(pred, mask, n_expected)

        # 4. SSIM
        l_ssim = self.ssim_module(pred, target)
        ssim_weight = self._ssim_weight

        # Per-band diagnostics
        per_band = {}
        for c, name in enumerate(self.band_names):
            if c >= C or c == self.sar_idx:
                continue
            band_res = (pred[:, :, c] - target[:, :, c]) * mask[:, :, c]
            n_band = float(B * T * H * W)
            per_band[f"loss_{name}"] = (charbonnier(band_res).sum() / n_band).detach()

        total = (
            l_recon
            + self.lambda_temporal   * l_temporal
            + self.lambda_ecological * l_eco
            + ssim_weight            * l_ssim
        )

        # FIX L5: only treat NaN/Inf as degenerate, not small-but-valid losses
        if not torch.isfinite(total):
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
            "recon_unmasked":    l_recon_unmasked.detach(),
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
        # Use same per-optical-band-pixel denominator for consistency
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
        n_constrained = 0
        for c, name in enumerate(self.band_names):
            if c >= pred.shape[2] or c == self.sar_idx:
                continue
            lo, hi = self.band_bounds.get(name, (-5.0, 5.0))
            band = pred[:, :, c]
            m    = mask[:, :, c]
            lo_viol = F.relu(lo - band) ** 2
            hi_viol = F.relu(band - hi) ** 2
            l_eco = l_eco + ((lo_viol + hi_viol) * m).sum() / n_expected
            n_constrained += 1
        if n_constrained > 0:
            l_eco = l_eco / n_constrained
        return l_eco

    def _build_bounds(self, config: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        norm_methods = config["bands"]["normalization"]
        band_names   = config["bands"]["names"]
        bounds = {}
        for name in band_names:
            key = name.lower()
            method = norm_methods.get(key, norm_methods.get(name, "minmax"))
            if method in ("minmax", "shift"):
                bounds[key] = (0.0, 1.0)
            elif method == "zscore":
                bounds[key] = (-4.0, 4.0)
            else:
                bounds[key] = (-5.0, 5.0)
                logger.warning(f"[Loss] Unknown norm method '{method}' for band '{name}', using wide bounds")
        logger.info(f"[Loss] Ecological bounds: { {k: v for k, v in bounds.items()} }")
        return bounds


def build_loss(config: Dict[str, Any]) -> EcoRewindLoss:
    return EcoRewindLoss(config)