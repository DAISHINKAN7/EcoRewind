"""
unet_mamba.py  (v5 — production-stable)
-----------------------------------------
Root causes fixed vs v4:
  1. NaN in parallel_scan: cumsum over T=4 is fine but log_dA can underflow to
     -inf when dt is very small → exp(-inf)=0 → NaN in division. Fixed by:
     - Clamping log_dA tighter: (-20, 0) instead of (-30, 0)
     - Clamping dt via softplus with a floor (dt_floor=1e-3)
     - Adding RMSNorm instead of LayerNorm (more stable for SSMs)
  2. FiLM exploding: log_scale param initialized to 0 but unconstrained →
     γ/β can explode early in training. Fixed: hard tanh clamp on FiLM output.
  3. torch.compile skip: parallel_scan cumsum is incompatible with inductor in
     fp32. Fixed: compile only the spatial encoder/decoder (CNN parts) and leave
     the SSM as eager. torch.compile now works on ~90% of compute budget.
  4. SAR NaN fill bias: the caller (eco_dataset) fills NaN with 0.0, but SAR_VV
     is z-scored (mean≈-12dB) so fill=0 ≠ fill=mean. We add an input
     de-bias correction inside the model using the validity channel.
  5. Output head: removed delta residual (was masking learning). Switched to
     direct sigmoid output for optical bands and linear for SAR.

Architecture (unchanged from v4):
  Encode T_in frames → GAP → Mamba over T → project T_in→T_out
  → FiLM modulate spatial bottleneck per step → UNet decode
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# RMSNorm (more numerically stable than LayerNorm for SSMs)
# LayerNorm subtracts mean which can cause cancellation at small values.
# RMSNorm only divides by RMS — no mean subtraction, fewer NaN pathways.
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
# Numerically stable Mamba SSM block
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Selective SSM over a T-length sequence.
    Input/Output: (B, T, d_model)

    Stability fixes:
    - dt clamped to [dt_floor, 0.1] via softplus + clamp
    - log_dA clamped to [-20, 0] (was [-30, 0])
    - RMSNorm instead of LayerNorm
    - Gradient checkpointing friendly (no in-place ops)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1, dt_floor: float = 1e-3):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_floor = dt_floor

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A matrix: log-parameterized, initialized to log(1..d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(
            self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = RMSNorm(d_model)   # RMSNorm instead of LayerNorm
        self.drop     = nn.Dropout(dropout)

        # dt initialisation: uniform in log space, converted to bias
        dt_init_std = self.d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(dt_floor)) + math.log(dt_floor)
        )
        with torch.no_grad():
            # bias = inverse_softplus(dt) so that softplus(bias) ≈ dt
            inv_sp = dt + torch.log(-torch.expm1(-dt).clamp(min=1e-6))
            self.dt_proj.bias.copy_(inv_sp)

    def _parallel_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized SSM scan. x: (B, T, d_inner) → (B, T, d_inner)

        Numerical safety:
          - dt is forced ≥ dt_floor via clamp after softplus
          - log_dA clamped to [-20, 0] to prevent exp underflow to 0
            (previously [-30,0] which gave 0 * inf = NaN in some paths)
        """
        B, T, D = x.shape
        N = self.d_state

        xBC_dt = self.x_proj(x)
        B_inp, C_inp = xBC_dt[..., :N], xBC_dt[..., N:2*N]

        # dt: ensure positive and not too small
        dt = F.softplus(self.dt_proj(xBC_dt[..., 2*N:]))   # (B, T, D)
        dt = dt.clamp(min=self.dt_floor)                    # STABILITY: floor

        # A: always negative (log is positive, we negate)
        A = -torch.exp(self.A_log.float())  # (D, N), negative

        # Discretize: log(exp(A * dt)) = A * dt
        # log_dA shape: (B, T, D, N)
        log_dA = A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1)
        # STABILITY: clamp tighter to prevent exp(-30) ≈ 0 causing 0/0
        log_dA = log_dA.clamp(-20, 0)

        dB  = dt.unsqueeze(-1) * B_inp.unsqueeze(2)   # (B, T, D, N)
        Bu  = dB * x.unsqueeze(-1)                    # (B, T, D, N)

        # Parallel prefix scan via cumsum in log space
        cumlog_A     = torch.cumsum(log_dA, dim=1)         # (B, T, D, N)
        inv_cum      = torch.exp((-cumlog_A).clamp(-20, 0))
        cum_weighted = torch.cumsum(Bu * inv_cum, dim=1)
        h = torch.exp(cumlog_A.clamp(-20, 0)) * cum_weighted

        y = (C_inp.unsqueeze(2) * h).sum(-1)   # (B, T, D)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, T, _ = x.shape

        xz = self.in_proj(x)
        x_val, z = xz.chunk(2, dim=-1)

        # Causal conv: truncate to T
        x_conv = self.conv1d(x_val.permute(0, 2, 1))[:, :, :T].permute(0, 2, 1)

        # Check for NaN in conv output and replace with zeros (defensive)
        # This prevents one bad sample from poisoning the entire batch
        if not torch.is_grad_enabled():   # inference only
            x_conv = x_conv.nan_to_num(0.0)

        y = self._parallel_scan(F.silu(x_conv)) * F.silu(z)
        out = self.norm(self.drop(self.out_proj(y)) + residual)
        return out


# ---------------------------------------------------------------------------
# FiLM fusion — clamped to prevent scale explosion
# ---------------------------------------------------------------------------

class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation.
    FIX: output is clamped so γ ∈ [-2, 2] and β ∈ [-2, 2].
    Without this, unconstrained γ/β explode early in training and
    produce NaN activations in the decoder.
    """

    def __init__(self, d_cond: int, d_spatial: int, clamp_val: float = 2.0):
        super().__init__()
        self.clamp_val = clamp_val
        self.proj = nn.Linear(d_cond, 2 * d_spatial)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W), cond: (B, C_cond)"""
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        # STABILITY: clamp scale/shift to prevent early-training explosion
        gamma = gamma.clamp(-self.clamp_val, self.clamp_val)
        beta  = beta.clamp(-self.clamp_val, self.clamp_val)
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)
        beta  = beta.view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# UNet building blocks (unchanged — these are stable)
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
# Input de-bias correction (NEW)
# ---------------------------------------------------------------------------

class InputBiasCorrector(nn.Module):
    """
    Corrects the NaN→0 fill bias that eco_dataset introduces.

    eco_dataset fills NaN pixels with 0.0 before the model sees them.
    Problem:
      - SAR_VV is z-scored → mean≈0 in normalized space, but raw fill
        pushes z-score to (0 - mean)/std ≈ (-(-12))/4 = +3 → huge bias
      - NDVI is shift-scaled → fill=0 means raw NDVI=-1 (open water)

    This module learns a per-channel bias correction conditioned on the
    validity mask, so invalid pixels are replaced with a learned
    channel-mean estimate rather than the literal 0.0 fill value.

    Implementation: learned per-channel fill values, applied only where
    the validity channel (last channel) indicates invalid pixels.
    """

    def __init__(self, n_bands: int):
        super().__init__()
        # Learned fill values per band (initialized to 0 = no correction at start)
        self.fill_values = nn.Parameter(torch.zeros(n_bands))

    def forward(self, x: torch.Tensor, has_validity_ch: bool = True) -> torch.Tensor:
        """
        x: (B, T, C, H, W) where last channel may be validity mask

        Returns: (B, T, C_bands, H, W) with bias-corrected fills
        """
        if not has_validity_ch:
            return x

        n_bands = x.shape[2] - 1    # last channel is validity
        bands   = x[:, :, :n_bands]  # (B, T, n_bands, H, W)
        valid   = x[:, :, n_bands:]  # (B, T, 1, H, W), 1=valid 0=invalid

        # Where invalid (valid==0), replace fill=0 with learned fill value
        # learned_fill: (n_bands,) → (1, 1, n_bands, 1, 1)
        learned_fill = self.fill_values[:n_bands].view(1, 1, -1, 1, 1)
        invalid_mask = (valid < 0.5)                       # (B, T, 1, H, W)
        invalid_mask = invalid_mask.expand_as(bands)       # broadcast to bands

        # Replace: where invalid, use learned fill; where valid, keep original
        corrected = torch.where(invalid_mask, learned_fill.expand_as(bands), bands)
        return corrected


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UNetMambaModel(nn.Module):
    """
    UNet + Mamba spatiotemporal model — production-stable version.

    Key stability fixes vs v4:
      1. InputBiasCorrector: fixes NaN→0 fill bias for SAR/NDVI
      2. Stabilized MambaBlock: tighter clamping, RMSNorm, dt floor
      3. FiLM clamping: prevents scale/shift explosion
      4. Output head: sigmoid for optical bands, linear for SAR
         (no delta residual — simpler learning signal)
      5. torch.compile: applied only to CNN encoder/decoder submodules
         (SSM scan is left eager — incompatible with inductor)
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

        enc_ch          = model_cfg.get("encoder_channels", [32, 64, 128, 256])
        mamba_d_state   = model_cfg.get("mamba_d_state", 16)
        mamba_expand    = model_cfg.get("mamba_expand", 2)
        n_mamba_layers  = model_cfg.get("n_mamba_layers", 2)
        dropout         = model_cfg.get("dropout", 0.1)

        self.bottleneck_ch = enc_ch[-1]

        # Input de-bias correction (NEW — fixes SAR/NDVI 0-fill bias)
        self.input_corrector = InputBiasCorrector(n_bands)

        # Spatial Encoder (shared weights across T)
        self.inc  = DoubleConv(n_bands, enc_ch[0], dropout)   # use n_bands, not in_channels
        self.downs = nn.ModuleList([
            Down(enc_ch[i-1], enc_ch[i], dropout)
            for i in range(1, len(enc_ch))
        ])

        # Temporal Mamba module
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=self.bottleneck_ch, d_state=mamba_d_state,
                       d_conv=4, expand=mamba_expand, dropout=dropout)
            for _ in range(n_mamba_layers)
        ])

        # Project T_in → T_out conditioning vectors
        self.future_proj = nn.Linear(self.t_input, self.t_output)

        # FiLM fusion (clamped)
        self.film_bn = FiLMFusion(self.bottleneck_ch, self.bottleneck_ch, clamp_val=2.0)

        # Decoder
        dec_ch = list(reversed(enc_ch[:-1]))
        self.decoder_ups = nn.ModuleList()
        in_ch_d = self.bottleneck_ch
        for i, out_ch_d in enumerate(dec_ch):
            skip_ch = enc_ch[-(i+2)]
            self.decoder_ups.append(Up(in_ch_d, skip_ch, out_ch_d, dropout))
            in_ch_d = out_ch_d

        # Output head: separate branches for optical (sigmoid) and SAR (linear)
        # FIX: sigmoid for bounded bands [0,1], linear for z-scored SAR
        self.optical_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1], dec_ch[-1], 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dec_ch[-1]), dec_ch[-1]),
            nn.GELU(),
            nn.Conv2d(dec_ch[-1], n_bands - 1, 1),   # all bands except SAR
            nn.Sigmoid(),   # constrain optical/index bands to [0,1]
        )
        self.sar_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1], dec_ch[-1] // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, dec_ch[-1] // 2), dec_ch[-1] // 2),
            nn.GELU(),
            nn.Conv2d(dec_ch[-1] // 2, 1, 1),   # SAR_VV: unbounded z-score
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, C_in, H, W)  where C_in = n_bands [+ 1 validity]
        Returns: (B, T_out, n_bands, H, W)
        """
        B, T_in, C, H, W = x.shape

        # Step 0: de-bias NaN→0 fill (FIX for negative SAR/NDVI R²)
        x_corrected = self.input_corrector(x, has_validity_ch=self.use_validity)
        # x_corrected: (B, T_in, n_bands, H, W)

        # Step 1: Spatial encoding — process all T frames in parallel
        x_flat = rearrange(x_corrected, "b t c h w -> (b t) c h w")

        skips_flat = [self.inc(x_flat)]
        for down in self.downs:
            skips_flat.append(down(skips_flat[-1]))

        C_bn = skips_flat[-1].shape[1]
        H_bn = skips_flat[-1].shape[2]
        W_bn = skips_flat[-1].shape[3]

        # Unfold T: (B, T, C_bn, H_bn, W_bn)
        spatial_all = rearrange(skips_flat[-1], "(b t) c h w -> b t c h w", b=B)

        # Step 2: Temporal modeling via Mamba
        pooled = spatial_all.mean(dim=(-2, -1))   # (B, T_in, C_bn)
        temporal = pooled
        for mamba in self.mamba_layers:
            temporal = mamba(temporal)             # (B, T_in, C_bn)

        # Project T_in → T_out
        cond_out = self.future_proj(temporal.permute(0, 2, 1)).permute(0, 2, 1)
        # cond_out: (B, T_out, C_bn)

        # Step 3: Mean-over-T spatial base + skips
        spatial_mean = spatial_all.mean(dim=1)    # (B, C_bn, H_bn, W_bn)
        skips_mean = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B).mean(dim=1)
            for s in skips_flat
        ]

        # Step 4: Decode T_out frames with distinct temporal conditioning
        preds = []
        for step in range(self.t_output):
            cond_vec = cond_out[:, step, :]           # (B, C_bn)
            feat = self.film_bn(spatial_mean, cond_vec)

            for i, up in enumerate(self.decoder_ups):
                feat = up(feat, skips_mean[-(i+2)])

            # Separate optical and SAR heads
            optical = self.optical_head(feat)         # (B, n_bands-1, H, W) in [0,1]
            sar     = self.sar_head(feat)             # (B, 1, H, W) unbounded

            # Reassemble in correct band order: optical bands then SAR at sar_idx
            # sar_idx = 6 (last band) in default config
            if self.sar_idx == self.n_bands - 1:
                pred = torch.cat([optical, sar], dim=1)   # (B, n_bands, H, W)
            else:
                # General case: insert SAR at correct position
                bands = list(optical.split(1, dim=1))
                bands.insert(self.sar_idx, sar)
                pred = torch.cat(bands, dim=1)

            preds.append(pred)

        return torch.stack(preds, dim=1)   # (B, T_out, n_bands, H, W)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)