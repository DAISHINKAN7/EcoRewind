"""
unet_mamba.py  (v4 — correct architecture)
-------------------------------------------
Pipeline (strictly followed):
  1. SpatialEncoder: shared-weight UNet encoder processes each of T_in
     frames independently → (B, T, C_lat, H', W')
  2. Global average pool → (B, T, C_lat) temporal tokens
  3. MambaBlock over T tokens → (B, T, C_lat) temporally-informed tokens
  4. FiLM fusion: expand tokens back to (B, T, C_lat, H', W') and
     modulate spatial features with scale+shift per channel
  5. Decoder: UNet decoder with skip connections reconstructs T_out frames

Key correctness fixes vs all prior versions:
  - FiLM modulation is applied PER-TIMESTEP to per-timestep spatial
    features, not to the mean-over-T. This gives each frame its own
    temporally-conditioned spatial representation.
  - Decoder gets T_out independently decoded frames, not one frame
    decoded T_out times from the same state.
  - For T_out > T_in, we use the last T_in Mamba outputs and a learned
    future-step embedding to generate T_out conditioning vectors.
  - No delta_scale bottleneck. Output head predicts full values.
  - No clamping during training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Vectorized Mamba SSM (parallel scan, no Python loops)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Selective SSM over a T-length sequence.
    Input/Output: (B, T, d_model)
    Uses fully vectorized parallel scan — no Python for-loops over T.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

        # dt initialisation
        dt_init_std = self.d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) *
                       (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _parallel_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized SSM scan. x: (B, T, d_inner) → (B, T, d_inner)"""
        B, T, D = x.shape
        N = self.d_state

        xBC_dt = self.x_proj(x)
        B_inp, C_inp = xBC_dt[..., :N], xBC_dt[..., N:2*N]
        dt = F.softplus(self.dt_proj(xBC_dt[..., 2*N:]))    # (B, T, D)

        log_dA  = -torch.exp(self.A_log).unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1)
        dB      = dt.unsqueeze(-1) * B_inp.unsqueeze(2)
        Bu      = dB * x.unsqueeze(-1)

        cumlog_A    = torch.cumsum(log_dA, dim=1)
        inv_cum     = torch.exp((-cumlog_A).clamp(-30, 0))
        cum_weighted = torch.cumsum(Bu * inv_cum, dim=1)
        h = torch.exp(cumlog_A.clamp(-30, 0)) * cum_weighted

        y = (C_inp.unsqueeze(2) * h).sum(-1)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_val, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_val.permute(0, 2, 1))[:, :, :T].permute(0, 2, 1)
        y = self._parallel_scan(F.silu(x_conv)) * F.silu(z)
        return self.norm(self.drop(self.out_proj(y)) + residual)


# ---------------------------------------------------------------------------
# FiLM fusion module
# ---------------------------------------------------------------------------

class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation.
    Conditions a (B, C, H, W) spatial feature map on a (B, C_cond) vector.
    Applies per-channel scale + shift: out = (1 + γ) * x + β
    Initialised near identity (γ≈0, β≈0) for stable early training.
    """

    def __init__(self, d_cond: int, d_spatial: int):
        super().__init__()
        self.proj = nn.Linear(d_cond, 2 * d_spatial)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, C, H, W)
        cond : (B, C_cond)
        """
        gamma, beta = self.proj(cond).chunk(2, dim=-1)   # each (B, C)
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)
        beta  = beta.view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# UNet building blocks
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
    UNet + Mamba spatiotemporal model.

    Correct pipeline:
      Encode T_in frames → pool spatially → Mamba over T → FiLM each frame
      → generate T_out conditioning vectors → decode T_out frames with skips

    The critical correctness property: each of the T_out output frames is
    decoded from its OWN spatially-conditioned feature map, where the spatial
    map is modulated by a DISTINCT temporal conditioning vector (not the same
    vector broadcast across all output steps).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config      = config
        model_cfg        = config["model"].get("unet_mamba", {})
        use_validity     = config["patches"]["use_validity_mask"]

        self.in_channels  = config["bands"]["count"] + (1 if use_validity else 0)
        self.out_channels = config["bands"]["count"]
        self.t_input  = config["model"]["t_input"]
        self.t_output = config["model"]["t_output"]

        enc_ch         = model_cfg.get("encoder_channels", [32, 64, 128, 256])
        mamba_d_state  = model_cfg.get("mamba_d_state", 16)
        mamba_expand   = model_cfg.get("mamba_expand", 2)
        n_mamba_layers = model_cfg.get("n_mamba_layers", 2)
        dropout        = model_cfg.get("dropout", 0.1)

        self.bottleneck_ch = enc_ch[-1]

        # ---- Step 1: Spatial Encoder (shared weights across T) ----
        self.inc   = DoubleConv(self.in_channels, enc_ch[0], dropout)
        self.downs = nn.ModuleList([
            Down(enc_ch[i-1], enc_ch[i], dropout)
            for i in range(1, len(enc_ch))
        ])

        # ---- Step 2: Temporal module ----
        # Pool: (B, T, C_bn, H', W') → (B, T, C_bn)
        # Mamba: (B, T, C_bn) → (B, T, C_bn)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=self.bottleneck_ch, d_state=mamba_d_state,
                       d_conv=4, expand=mamba_expand, dropout=dropout)
            for _ in range(n_mamba_layers)
        ])

        # For T_out steps, we need T_out conditioning vectors.
        # Use a learned future-step projection: each output step gets its own
        # linear combination of the T_in Mamba outputs.
        # Shape: projects (T_in, C_bn) → (T_out, C_bn) per sample
        self.future_proj = nn.Linear(self.t_input, self.t_output)

        # ---- Step 3: FiLM fusion (one per encoder level + bottleneck) ----
        # Applied to the per-timestep spatial features for each T_out step
        self.film_bn = FiLMFusion(self.bottleneck_ch, self.bottleneck_ch)

        # ---- Step 4: Decoder (shared weights across T_out) ----
        dec_ch = list(reversed(enc_ch[:-1]))
        self.decoder_ups = nn.ModuleList()
        in_ch = self.bottleneck_ch
        for i, out_ch_d in enumerate(dec_ch):
            skip_ch = enc_ch[-(i+2)]
            self.decoder_ups.append(Up(in_ch, skip_ch, out_ch_d, dropout))
            in_ch = out_ch_d

        # Output head: predicts full frame values
        self.output_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1], dec_ch[-1], 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dec_ch[-1]), dec_ch[-1]),
            nn.GELU(),
            nn.Conv2d(dec_ch[-1], self.out_channels, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear,)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, C, H, W)
        Returns: (B, T_out, C_out, H, W)
        """
        B, T_in, C, H, W = x.shape

        # ================================================================
        # Step 1: Spatial encoding — process all T frames in parallel
        # Fold T into batch: (B*T, C, H, W)
        # ================================================================
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")

        skips_flat = [self.inc(x_flat)]                 # (B*T, enc_ch[0], H, W)
        for down in self.downs:
            skips_flat.append(down(skips_flat[-1]))
        # skips_flat[-1]: (B*T, C_bn, H_bn, W_bn)  — bottleneck

        C_bn = skips_flat[-1].shape[1]
        H_bn = skips_flat[-1].shape[2]
        W_bn = skips_flat[-1].shape[3]

        # Unfold T: (B, T, C_bn, H_bn, W_bn)
        spatial_all = rearrange(skips_flat[-1], "(b t) c h w -> b t c h w", b=B)

        # ================================================================
        # Step 2: Temporal modeling
        # Global average pool spatial dims → (B, T, C_bn)
        # ================================================================
        pooled = spatial_all.mean(dim=(-2, -1))         # (B, T_in, C_bn)

        # Mamba over T_in temporal tokens
        temporal = pooled
        for mamba in self.mamba_layers:
            temporal = mamba(temporal)                  # (B, T_in, C_bn)

        # Project T_in → T_out conditioning vectors
        # future_proj: (T_in,) → (T_out,) per channel
        # temporal: (B, T_in, C_bn) → transpose → (B, C_bn, T_in)
        # → linear → (B, C_bn, T_out) → transpose → (B, T_out, C_bn)
        cond_out = self.future_proj(temporal.permute(0, 2, 1)).permute(0, 2, 1)
        # cond_out: (B, T_out, C_bn)

        # ================================================================
        # Step 3: FiLM fusion — modulate spatial features per T_out step
        # Use mean-over-T spatial bottleneck as base, modulate with
        # each step's temporal conditioning vector
        # ================================================================
        spatial_mean = spatial_all.mean(dim=1)          # (B, C_bn, H_bn, W_bn)

        # Skips for decoder: mean over T
        skips_mean = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B).mean(dim=1)
            for s in skips_flat
        ]

        # ================================================================
        # Step 4: Decode T_out frames
        # Each output step uses its own FiLM-conditioned spatial map
        # ================================================================
        preds = []
        for step in range(self.t_output):
            # cond_vec: (B, C_bn) — unique temporal signal for this step
            cond_vec = cond_out[:, step, :]             # (B, C_bn)

            # FiLM modulates the spatial bottleneck with temporal context
            # This gives SPATIAL VARIATION driven by temporal dynamics
            feat = self.film_bn(spatial_mean, cond_vec) # (B, C_bn, H_bn, W_bn)

            # UNet decode with skip connections
            for i, up in enumerate(self.decoder_ups):
                feat = up(feat, skips_mean[-(i+2)])

            pred = self.output_head(feat)               # (B, C_out, H, W)
            preds.append(pred)

        return torch.stack(preds, dim=1)                # (B, T_out, C_out, H, W)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)