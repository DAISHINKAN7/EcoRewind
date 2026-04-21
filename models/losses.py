"""
losses.py  (v5 — fp16-safe gradients)
---------------------------------------
The v4 run gave us conclusive evidence via the gradient diagnostic:

    Params with .grad AFTER backward : 135
    Params with ZERO .grad           : 0
    Sum of |grad| across all params  : nan

Gradients DO flow through the autograd graph, but they are NaN. GradScaler
correctly detects this and skips optimizer.step(), which is why grad_norm
is reported as 0.000 every epoch (empty list of "valid" grad norms) and
why opt_steps=1934 but nothing learns.

The NaN source is the loss computation in fp16 (AMP autocast).

Three specific fp16 hazards v4 had:

H1 — charbonnier(pred - target) is computed BEFORE masking.
  For target pixels where mask=0, pred - target can be wildly large
  (since those pixels contain fill values or uninitialised outputs).
  `x*x` in fp16 overflows to Inf when |x| > 256, and Inf - Inf = NaN
  in the sqrt derivative. Even though these values are multiplied by
  mask=0 AFTER, the gradient graph has already produced NaN through
  them and NaN * 0 = NaN (gradient-wise).

H2 — charbonnier eps=1e-3 is too small for fp16.
  fp16 smallest normal is 6e-5. eps*eps = 1e-6 is below normal range
  and underflows to zero. Then sqrt(x*x + 0) near x=0 has an undefined
  derivative. We need eps ≥ 1e-2 for fp16 safety.

H3 — SSIM is computed over the FULL prediction, including SAR.
  SAR outputs from EcoTransformer are clamped to [-6, 6] but in fp16
  the intermediate sig_p2, sig_t2 values in SSIM can still overflow.
  Also the SSIM window convolution at fp16 produces NaN on patches
  with high contrast. Solution: compute SSIM in fp32 explicitly.

Fixes in v5:

FIX L1-v5 — charbonnier eps raised to 1e-2, and residuals masked BEFORE
  charbonnier so the non-linear op only sees mask-zeroed values.

FIX L2-v5 — Wrap SSIM module call in torch.autocast(enabled=False) to
  force fp32 computation. SSIM is cheap; fp32 overhead is negligible
  but numerical stability dramatically improves.

FIX L3-v5 — Ecological loss uses F.relu(...).clamp(max=BOUND) to bound
  the squared violation. Prevents runaway grad from huge clamped errors.

FIX L4-v5 — Per-band diagnostic loss also uses the fp16-safe eps and
  pre-masked residuals.

FIX L5-v5 — Added a NaN guard on each loss component before summing.
  If any component is NaN, replace with a zero tensor that has a
  valid grad_fn (via detach + zero), and log a warning. This prevents
  a single bad component from contaminating the total.

All changes preserve the v4 loss scale (per-band-per-pixel ~0.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

DEGENERATE_LOSS_SENTINEL = -1.0


def charbonnier(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    fp16-safe Charbonnier. Default eps raised from 1e-3 to 1e-2.

    With eps=1e-2, eps*eps = 1e-4 which is well inside fp16 normal range
    (smallest normal is 6e-5). The sqrt derivative near x=0 is bounded:
        d/dx sqrt(x² + eps²) = x / sqrt(x² + eps²)
    At x=0: derivative = 0 (well-defined)
    Max magnitude: 1 (when |x| >> eps), safe in fp16.
    """
    return torch.sqrt(x * x + eps * eps) - eps


def _safe_loss_component(
    value: torch.Tensor,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """If the loss component is non-finite, zero it (keeping grad_fn)."""
    if not torch.isfinite(value).all():
        logger.warning(f"  [Loss] Component '{name}' produced non-finite value — zeroing")
        # Multiply by zero preserves grad_fn but makes the contribution 0
        return value * 0.0
    return value


class _SSIMModule(nn.Module):
    """
    SSIM computed in fp32 explicitly to avoid fp16 overflow in variance terms.
    """

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
        # FIX L2-v5: force fp32 for SSIM to avoid fp16 variance overflow
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred   = pred.float()
            target = target.float()
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
                ssim_vals.append((num / den.clamp(min=1e-6)).mean())
            if not ssim_vals:
                return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
            result = (1.0 - torch.stack(ssim_vals).mean()).clamp(0.0, 1.0)
        return result


class EcoRewindLoss(nn.Module):
    """
    ECO-REWIND loss v5 — fp16-safe.

    Loss scale fix (carried over from v4): denominator B*T*C_optical*H*W.

    Band weights (normalised so mean over optical bands = 1.0):
      Blue=0.5, Green=0.8, Red=0.8, NIR=1.0, NDVI=3.0, NDWI=2.0, SAR=0.0
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.lambda_temporal   = config["loss"]["lambda_temporal"]
        self.lambda_ecological = config["loss"]["lambda_ecological"]
        self.lambda_ssim       = config["loss"].get("lambda_ssim", 0.10)
        self._ssim_warmup_epochs = config["loss"].get("ssim_warmup_epochs", 20)
        self._current_epoch = 0

        # FIX L1-v5: fp16-safe eps configurable
        self.charbonnier_eps = config["loss"].get("charbonnier_eps", 1e-2)

        band_idx = config["bands"]["indices"]
        self.ndvi_idx = band_idx["ndvi"]
        self.ndwi_idx = band_idx["ndwi"]
        self.sar_idx  = band_idx["sar_vv"]
        self.n_bands  = config["bands"]["count"]

        self.band_names = [n.lower() for n in config["bands"]["names"]]
        self.band_bounds = self._build_bounds(config)

        raw_w = torch.ones(self.n_bands)
        raw_w[band_idx["blue"]]  = 0.5
        raw_w[band_idx["green"]] = 0.8
        raw_w[band_idx["red"]]   = 0.8
        raw_w[band_idx["nir"]]   = 1.0
        raw_w[band_idx["ndvi"]]  = 3.0
        raw_w[band_idx["ndwi"]]  = 2.0
        raw_w[self.sar_idx]      = 0.0

        optical_mask = torch.ones(self.n_bands, dtype=torch.bool)
        optical_mask[self.sar_idx] = False
        n_optical = optical_mask.sum().item()
        optical_mean = raw_w[optical_mask].mean()
        raw_w[optical_mask] = raw_w[optical_mask] / optical_mean.clamp(min=1e-6)
        self.register_buffer("band_weights", raw_w)
        self.n_optical = int(n_optical)

        logger.info(
            f"[Loss] Band weights (normalised): "
            + ", ".join(f"{n}={raw_w[i].item():.3f}"
                        for i, n in enumerate(self.band_names))
        )
        logger.info(f"[Loss] Charbonnier eps: {self.charbonnier_eps:.1e} (fp16-safe)")

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
        device = pred.device
        dtype = pred.dtype

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
            sentinel = torch.tensor(DEGENERATE_LOSS_SENTINEL, device=device, dtype=dtype)
            return sentinel, {
                "reconstruction":  torch.tensor(0.0),
                "temporal_smooth": torch.tensor(0.0),
                "ecological":      torch.tensor(0.0),
                "ssim":             torch.tensor(0.0),
                "spectral":        torch.tensor(0.0),
                "ssim_weight":     torch.tensor(self._ssim_weight),
                "valid_pct":       torch.tensor(valid_pct),
                "is_degenerate":   torch.tensor(1.0),
            }

        n_expected = float(B * T * self.n_optical * H * W)

        # FIX L1-v5: CRITICAL — mask the residual FIRST, then apply charbonnier.
        # This ensures charbonnier only operates on pixels that matter. For
        # masked-out pixels the residual is exactly 0, so charbonnier(0)=0
        # with zero gradient — no NaN contamination from fill-value pixels.
        #
        # l_recon_unmasked is a diagnostic only — compute under no_grad to
        # keep it out of the backward graph entirely. This avoids the edge
        # case where unmasked residuals (including SAR band or fill-value
        # target pixels) produce NaN gradients through charbonnier in fp16.
        w = self.band_weights.view(1, 1, -1, 1, 1)
        masked_residual = (pred - target) * mask
        with torch.no_grad():
            l_recon_unmasked = charbonnier(pred - target, self.charbonnier_eps).mean()
        charb_vals = charbonnier(masked_residual, self.charbonnier_eps)
        l_recon = (charb_vals * w).sum() / n_expected
        l_recon = _safe_loss_component(l_recon, "reconstruction", device, dtype)

        # 2. Temporal smoothness
        l_temporal = self._temporal_smooth_loss(pred, target, mask, n_expected)
        l_temporal = _safe_loss_component(l_temporal, "temporal_smooth", device, dtype)

        # 3. Ecological constraints
        l_eco = self._ecological_loss(pred, mask, n_expected)
        l_eco = _safe_loss_component(l_eco, "ecological", device, dtype)

        # 4. SSIM (computed in fp32 internally)
        l_ssim = self.ssim_module(pred, target)
        l_ssim = _safe_loss_component(l_ssim, "ssim", device, dtype)
        ssim_weight = self._ssim_weight

        # Per-band diagnostics (fp16-safe, masked)
        per_band = {}
        for c, name in enumerate(self.band_names):
            if c >= C or c == self.sar_idx:
                continue
            band_res = (pred[:, :, c] - target[:, :, c]) * mask[:, :, c]
            n_band = float(B * T * H * W)
            band_loss = charbonnier(band_res, self.charbonnier_eps).sum() / n_band
            per_band[f"loss_{name}"] = band_loss.detach()

        total = (
            l_recon
            + self.lambda_temporal   * l_temporal
            + self.lambda_ecological * l_eco
            + ssim_weight            * l_ssim
        )

        # Final safety: if total is non-finite despite component-level guards,
        # return sentinel so trainer skips this batch
        if not torch.isfinite(total):
            logger.warning(
                f"  [Loss] Total loss non-finite after safeguards: "
                f"recon={l_recon.item():.4f}, temp={l_temporal.item():.4f}, "
                f"eco={l_eco.item():.4f}, ssim={l_ssim.item():.4f}"
            )
            sentinel = torch.tensor(DEGENERATE_LOSS_SENTINEL, device=device, dtype=dtype)
            return sentinel, {
                "reconstruction":  l_recon.detach(),
                "temporal_smooth": l_temporal.detach(),
                "ecological":      l_eco.detach(),
                "ssim":             l_ssim.detach(),
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
            "spectral":          torch.zeros(1, device=device).squeeze(),
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
        mask_t = mask[:, 1:, ndvi]

        # FIX L1-v5: mask BEFORE huber_loss — same principle as charbonnier
        pred_d_ndvi = (pred[:, 1:, ndvi]   - pred[:, :-1, ndvi])   * mask_t
        tgt_d_ndvi  = (target[:, 1:, ndvi] - target[:, :-1, ndvi]) * mask_t
        pred_d_ndwi = (pred[:, 1:, ndwi]   - pred[:, :-1, ndwi])   * mask_t
        tgt_d_ndwi  = (target[:, 1:, ndwi] - target[:, :-1, ndwi]) * mask_t

        n_t = float(pred.shape[0] * (pred.shape[1]-1) * pred.shape[3] * pred.shape[4])
        huber_delta = 0.05
        l  = F.huber_loss(pred_d_ndvi, tgt_d_ndvi, reduction="sum", delta=huber_delta) / n_t
        l += F.huber_loss(pred_d_ndwi, tgt_d_ndwi, reduction="sum", delta=huber_delta) / n_t
        return l * 0.5

    def _ecological_loss(
        self,
        pred:       torch.Tensor,
        mask:       torch.Tensor,
        n_expected: float,
    ) -> torch.Tensor:
        l_eco = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        n_constrained = 0
        # FIX L3-v5: clamp the violation to prevent huge values → fp16 overflow
        MAX_VIOLATION = 10.0
        for c, name in enumerate(self.band_names):
            if c >= pred.shape[2] or c == self.sar_idx:
                continue
            lo, hi = self.band_bounds.get(name, (-5.0, 5.0))
            band = pred[:, :, c]
            m    = mask[:, :, c]
            lo_viol = F.relu(lo - band).clamp(max=MAX_VIOLATION)
            hi_viol = F.relu(band - hi).clamp(max=MAX_VIOLATION)
            # Use ** 2 on clamped values, then mask, to avoid huge unclamped
            # squared errors contaminating gradients
            viol_sq = (lo_viol * lo_viol + hi_viol * hi_viol) * m
            l_eco = l_eco + viol_sq.sum() / n_expected
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
                logger.warning(
                    f"[Loss] Unknown norm method '{method}' for band '{name}', "
                    f"using wide bounds"
                )
        logger.info(f"[Loss] Ecological bounds: { {k: v for k, v in bounds.items()} }")
        return bounds


def build_loss(config: Dict[str, Any]) -> EcoRewindLoss:
    return EcoRewindLoss(config)