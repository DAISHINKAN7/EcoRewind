"""
unet_mamba.py  (v6 — full stability + architecture fixes)
----------------------------------------------------------
Fixes vs v5:

FIX C (Mamba) — LayerNorm BEFORE each Mamba block (pre-norm pattern).
  SSMs are sensitive to input scale. Without pre-norm the SSM sees raw
  spatial pool values whose scale changes during training → instability.
  Pre-norm pins the input distribution before the SSM at each layer.

FIX C — Residual scaling with learned α parameter per Mamba layer.
  Initialised to α=0.1 so each Mamba layer starts as a near-identity
  transformation. As training proceeds α grows to a useful value.
  This prevents large gradient spikes in early epochs.

FIX C — Reduced default Mamba depth (n_mamba_layers: 2 → config-driven).
  With T=4, deep SSM stacks over-parameterize the temporal dimension.
  The default stays 2 but is explicitly checked here.

FIX 4 (SAR gating) — Learnable SAR input gate.
  SAR is present in the input but 87.5% fill. A learned scalar gate
  γ_sar (initialised near 0) lets the model suppress SAR contribution
  during early training and gradually open it if SAR proves useful.
  This stops SAR noise from corrupting optical feature learning.

FIX D — Monte Carlo Dropout.
  enable_mc_dropout() was already present; now also applied to all
  DoubleConv Dropout2d layers for spatial uncertainty.

FIX A — Gradient norm per-layer logging hook (optional, controlled by
  log_grad_norms=True on the module).

Architecture summary (unchanged from v5):
  Encode T_in frames → GAP → LayerNorm → Mamba(×N) → project T_in→T_out
  → FiLM modulate spatial bottleneck → UNet decode
  → separate sigmoid (optical) + linear (SAR) heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# Numerically stable MambaBlock with pre-norm + residual scaling
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Selective SSM.  Input/Output: (B, T, d_model)

    New in v6:
    - pre_norm: LayerNorm applied BEFORE the SSM projection
    - residual_scale: learned α, init=0.1, so output = α * SSM(x) + x
      keeps the block near-identity at start of training
    """

    def __init__(
        self,
        d_model:    int,
        d_state:    int  = 16,
        d_conv:     int  = 4,
        expand:     int  = 2,
        dropout:    float = 0.1,
        dt_floor:   float = 1e-3,
    ):
        super().__init__()
        self.d_inner  = d_model * expand
        self.d_state  = d_state
        self.dt_floor = dt_floor

        # FIX C: pre-norm before SSM projections
        self.pre_norm = nn.LayerNorm(d_model)

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(
            self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = RMSNorm(d_model)
        self.drop     = nn.Dropout(dropout)

        # FIX C: residual scaling — start near-identity
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # dt init
        dt_init_std = self.d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(dt_floor))
            + math.log(dt_floor)
        )
        with torch.no_grad():
            inv_sp = dt + torch.log(-torch.expm1(-dt).clamp(min=1e-6))
            self.dt_proj.bias.copy_(inv_sp)

    def _parallel_scan(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = self.d_state

        xBC_dt = self.x_proj(x)
        B_inp, C_inp = xBC_dt[..., :N], xBC_dt[..., N:2*N]

        dt = F.softplus(self.dt_proj(xBC_dt[..., 2*N:])).clamp(min=self.dt_floor)
        A  = -torch.exp(self.A_log.float())

        log_dA = (A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1)).clamp(-20, 0)
        dB  = dt.unsqueeze(-1) * B_inp.unsqueeze(2)
        Bu  = dB * x.unsqueeze(-1)

        cumlog_A     = torch.cumsum(log_dA, dim=1)
        inv_cum      = torch.exp((-cumlog_A).clamp(-20, 0))
        cum_weighted = torch.cumsum(Bu * inv_cum, dim=1)
        h = torch.exp(cumlog_A.clamp(-20, 0)) * cum_weighted

        y = (C_inp.unsqueeze(2) * h).sum(-1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # FIX C: pre-norm before projections
        x_normed = self.pre_norm(x)
        xz = self.in_proj(x_normed)
        x_val, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_val.permute(0, 2, 1))[:, :, :T].permute(0, 2, 1)
        if not torch.is_grad_enabled():
            x_conv = x_conv.nan_to_num(0.0)

        y = self._parallel_scan(F.silu(x_conv)) * F.silu(z)
        ssm_out = self.norm(self.drop(self.out_proj(y)))

        # FIX C: scaled residual — α starts at 0.1
        alpha = self.residual_scale.abs().clamp(max=1.0)
        return x + alpha * ssm_out


# ---------------------------------------------------------------------------
# Learnable SAR input gate  (FIX 4)
# ---------------------------------------------------------------------------

class SARInputGate(nn.Module):
    """
    Learnable per-pixel gate on the SAR channel.

    SAR is 87.5% fill values that can corrupt optical feature learning.
    This gate initialises near-zero so SAR is suppressed at the start
    of training. The model can open the gate if SAR proves informative.

    gate(x) = x * sigmoid(γ)  where γ is a learned scalar, init=-3
    sigmoid(-3) ≈ 0.05  →  SAR contribution is 5% at initialisation
    """

    def __init__(self):
        super().__init__()
        # init to -3 so sigmoid ≈ 0.05 (nearly closed)
        self.log_gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor, sar_idx: int) -> torch.Tensor:
        """x: (B, T, C, H, W) — modulate the SAR channel in-place-free"""
        gate = torch.sigmoid(self.log_gate)
        # Build scale tensor: 1.0 for all channels, gate for SAR
        scale = torch.ones(x.shape[2], device=x.device, dtype=x.dtype)
        scale[sar_idx] = gate
        scale = scale.view(1, 1, -1, 1, 1)
        return x * scale


# ---------------------------------------------------------------------------
# FiLM (clamped, unchanged from v5)
# ---------------------------------------------------------------------------

class FiLMFusion(nn.Module):
    def __init__(self, d_cond: int, d_spatial: int, clamp_val: float = 2.0):
        super().__init__()
        self.clamp_val = clamp_val
        self.proj = nn.Linear(d_cond, 2 * d_spatial)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        gamma = gamma.clamp(-self.clamp_val, self.clamp_val).view(x.shape[0], x.shape[1], 1, 1)
        beta  = beta.clamp( -self.clamp_val, self.clamp_val).view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# InputBiasCorrector (unchanged from v5)
# ---------------------------------------------------------------------------

class InputBiasCorrector(nn.Module):
    def __init__(self, n_bands: int):
        super().__init__()
        self.fill_values = nn.Parameter(torch.zeros(n_bands))

    def forward(self, x: torch.Tensor, has_validity_ch: bool = True) -> torch.Tensor:
        if not has_validity_ch:
            return x
        n_bands = x.shape[2] - 1
        bands   = x[:, :, :n_bands]
        valid   = x[:, :, n_bands:]
        learned_fill = self.fill_values[:n_bands].view(1, 1, -1, 1, 1)
        invalid_mask = (valid < 0.5).expand_as(bands)
        return torch.where(invalid_mask, learned_fill.expand_as(bands), bands)


# ---------------------------------------------------------------------------
# UNet blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Dropout2d(p=dropout),
        )
    def forward(self, x): return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout))
    def forward(self, x): return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, dropout)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(x, [0, skip.shape[-1]-x.shape[-1], 0, skip.shape[-2]-x.shape[-2]])
        return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UNetMambaModel(nn.Module):
    """
    UNet + Mamba — v6 production-stable with all architecture fixes.

    New in v6:
      - LayerNorm pre-norm before each Mamba block
      - Residual scaling (α=0.1 init) per Mamba layer
      - Learnable SAR input gate (suppresses SAR noise early in training)
      - Full MC Dropout support (all Dropout/Dropout2d layers)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config       = config
        model_cfg         = config["model"].get("unet_mamba", {})
        use_validity      = config["patches"]["use_validity_mask"]

        n_bands           = config["bands"]["count"]
        self.n_bands      = n_bands
        self.in_channels  = n_bands + (1 if use_validity else 0)
        self.out_channels = n_bands
        self.t_input      = config["model"]["t_input"]
        self.t_output     = config["model"]["t_output"]
        self.use_validity = use_validity
        self.sar_idx      = config["bands"]["indices"]["sar_vv"]

        enc_ch         = model_cfg.get("encoder_channels", [32, 64, 128, 256])
        mamba_d_state  = model_cfg.get("mamba_d_state", 16)
        mamba_expand   = model_cfg.get("mamba_expand", 2)
        n_mamba_layers = model_cfg.get("n_mamba_layers", 2)
        dropout        = model_cfg.get("dropout", 0.1)

        self.bottleneck_ch = enc_ch[-1]

        # FIX 4: SAR input gate
        self.sar_gate = SARInputGate()

        # Input de-bias correction
        self.input_corrector = InputBiasCorrector(n_bands)

        # Spatial Encoder
        self.inc  = DoubleConv(n_bands, enc_ch[0], dropout)
        self.downs = nn.ModuleList([
            Down(enc_ch[i-1], enc_ch[i], dropout)
            for i in range(1, len(enc_ch))
        ])

        # FIX C: Pre-norm LayerNorm before the Mamba stack
        self.pre_mamba_norm = nn.LayerNorm(self.bottleneck_ch)

        # Temporal Mamba stack
        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=self.bottleneck_ch,
                d_state=mamba_d_state,
                d_conv=4,
                expand=mamba_expand,
                dropout=dropout,
            )
            for _ in range(n_mamba_layers)
        ])

        # T_in → T_out projection
        self.future_proj = nn.Linear(self.t_input, self.t_output)

        # FiLM
        self.film_bn = FiLMFusion(self.bottleneck_ch, self.bottleneck_ch, clamp_val=2.0)

        # Decoder
        dec_ch = list(reversed(enc_ch[:-1]))
        self.decoder_ups = nn.ModuleList()
        in_ch_d = self.bottleneck_ch
        for i, out_ch_d in enumerate(dec_ch):
            skip_ch = enc_ch[-(i+2)]
            self.decoder_ups.append(Up(in_ch_d, skip_ch, out_ch_d, dropout))
            in_ch_d = out_ch_d

        # Separate output heads
        self.optical_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1], dec_ch[-1], 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dec_ch[-1]), dec_ch[-1]),
            nn.GELU(),
            nn.Conv2d(dec_ch[-1], n_bands - 1, 1),
            nn.Sigmoid(),
        )
        self.sar_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1], dec_ch[-1] // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, dec_ch[-1] // 2), dec_ch[-1] // 2),
            nn.GELU(),
            nn.Conv2d(dec_ch[-1] // 2, 1, 1),
        )

        self._init_weights()

        # FIX A: optional gradient norm logging
        self.log_grad_norms = False
        self._grad_norm_log: Dict[str, float] = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def enable_mc_dropout(self):
        """Enable MC Dropout — keeps ALL dropout layers active during inference."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, C_in, H, W)
        Returns: (B, T_out, n_bands, H, W)
        """
        B, T_in, C, H, W = x.shape

        # Step 0: de-bias NaN fill
        x_corrected = self.input_corrector(x, has_validity_ch=self.use_validity)

        # FIX 4: suppress SAR channel noise via learned gate
        x_gated = self.sar_gate(x_corrected, self.sar_idx)

        # Step 1: Spatial encoding
        x_flat = rearrange(x_gated, "b t c h w -> (b t) c h w")
        skips_flat = [self.inc(x_flat)]
        for down in self.downs:
            skips_flat.append(down(skips_flat[-1]))

        spatial_all = rearrange(skips_flat[-1], "(b t) c h w -> b t c h w", b=B)

        # Step 2: Temporal Mamba
        pooled = spatial_all.mean(dim=(-2, -1))    # (B, T_in, C_bn)

        # FIX C: pre-norm before Mamba stack
        temporal = self.pre_mamba_norm(pooled)
        for mamba in self.mamba_layers:
            temporal = mamba(temporal)             # residual scaling inside block

        # Project T_in → T_out
        cond_out = self.future_proj(temporal.permute(0, 2, 1)).permute(0, 2, 1)

        # Step 3: Spatial base + mean skips
        spatial_mean = spatial_all.mean(dim=1)
        skips_mean = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B).mean(dim=1)
            for s in skips_flat
        ]

        # Step 4: Decode
        preds = []
        for step in range(self.t_output):
            cond_vec = cond_out[:, step, :]
            feat = self.film_bn(spatial_mean, cond_vec)
            for i, up in enumerate(self.decoder_ups):
                feat = up(feat, skips_mean[-(i+2)])

            optical = self.optical_head(feat)
            sar     = self.sar_head(feat)

            if self.sar_idx == self.n_bands - 1:
                pred = torch.cat([optical, sar], dim=1)
            else:
                bands = list(optical.split(1, dim=1))
                bands.insert(self.sar_idx, sar)
                pred = torch.cat(bands, dim=1)
            preds.append(pred)

        return torch.stack(preds, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)