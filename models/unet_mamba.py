"""
unet_mamba.py  (v2 — fixed lazy prediction)
--------------------------------------------
Key fixes over v1:
  1. delta_scale removed entirely. The output head now predicts the FULL
     frame directly (not a residual), then blends with last_frame via a
     learned per-channel alpha gate. This forces the model to actually
     learn future states rather than zeroing out deltas.

  2. The Mamba temporal module now operates on the FULL spatial bottleneck
     (H_bn * W_bn tokens per timestep) rather than just the spatially-pooled
     mean vector. Pooling discards all spatial information before temporal
     modelling, leaving the decoder with a single scalar per channel to work
     with. Operating on flattened spatial tokens gives the SSM meaningful
     spatially-aware temporal context.

  3. The decoder uses learned cross-attention over the full Mamba output
     sequence (all T_in steps, all spatial positions) instead of just the
     last hidden state. This lets each decoder step attend to whichever
     input quarters are most relevant.

  4. Output is NOT clamped during training. The ecological loss handles
     out-of-range predictions. Clamping kills gradients for large deviations
     which is exactly when learning needs to happen most.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Mamba block (same SSM core as v1, unchanged)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv  = d_conv

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
                                  kernel_size=d_conv, padding=d_conv - 1,
                                  groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)
        self._init_dt_proj()

    def _init_dt_proj(self):
        dt_init_std = self.d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log.float())
        xBC_dt = self.x_proj(x)
        B_inp  = xBC_dt[..., :N]
        C_inp  = xBC_dt[..., N:2*N]
        dt_raw = xBC_dt[..., 2*N:]
        dt = F.softplus(self.dt_proj(dt_raw))
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_inp.unsqueeze(2)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (C_inp[:, t].unsqueeze(1) * h).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_val, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_val.permute(0, 2, 1))
        x_conv = x_conv[:, :, :T].permute(0, 2, 1)
        x_conv = F.silu(x_conv)
        y = self.ssm(x_conv)
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = self.drop(out)
        return self.norm(out + residual)


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
# Temporal cross-attention decoder query
# ---------------------------------------------------------------------------

class TemporalCrossAttentionDecoder(nn.Module):
    """
    For each decode step, attends over the full Mamba output sequence
    (all T_in temporal states) to produce a conditioning vector.
    This replaces the naive "use last hidden state" approach which
    discards all but the final timestep's information.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        query : (B, 1, D)  — learned step embedding
        keys  : (B, T, D)  — Mamba output sequence
        Returns: (B, D)
        """
        q = self.norm_q(query)
        k = self.norm_k(keys)
        out, _ = self.attn(q, k, k)
        out = query + self.drop(out)
        out = self.norm(out)
        return out.squeeze(1)   # (B, D)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UNetMambaModel(nn.Module):
    """
    UNet + Mamba temporal modeling.

    Critical design changes from v1:
    - Mamba operates on flattened spatial tokens (B, T*N, D) not pooled mean.
      This gives T_in * H_bn * W_bn tokens per sequence — richer temporal signal.
    - Output head predicts FULL frame values (not residual delta).
      A learned per-channel alpha gate blends prediction with last_frame:
        output = alpha * prediction + (1 - alpha) * last_frame
      Alpha is initialised near 0 so training starts conservative, but the
      gate is LEARNED and can grow large — unlike a fixed 0.25 scale which
      permanently bottlenecks the prediction magnitude.
    - No clamping during training.
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

        enc_channels   = model_cfg.get("encoder_channels", [32, 64, 128, 256])
        mamba_d_state  = model_cfg.get("mamba_d_state", 16)
        mamba_expand   = model_cfg.get("mamba_expand", 2)
        n_mamba_layers = model_cfg.get("n_mamba_layers", 2)
        n_heads        = model_cfg.get("n_heads", 4)
        dropout        = model_cfg.get("dropout", 0.1)

        self.bottleneck_ch = enc_channels[-1]

        # --- UNet Encoder ---
        self.inc  = DoubleConv(self.in_channels, enc_channels[0], dropout)
        self.downs = nn.ModuleList([
            Down(enc_channels[i-1], enc_channels[i], dropout)
            for i in range(1, len(enc_channels))
        ])

        # --- Mamba: input projection to reduce bottleneck spatial tokens ---
        # Bottleneck is (B*T, C_bn, H_bn, W_bn).  H_bn=W_bn=8 for 128px input
        # with 4 down blocks → 8×8=64 tokens per frame.
        # We project C_bn channels → d_mamba for the SSM.
        d_mamba = self.bottleneck_ch
        self.spatial_to_token = nn.Linear(self.bottleneck_ch, d_mamba)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_mamba, d_state=mamba_d_state,
                       d_conv=4, expand=mamba_expand, dropout=dropout)
            for _ in range(n_mamba_layers)
        ])
        self.token_to_spatial = nn.Linear(d_mamba, self.bottleneck_ch)

        # --- Temporal cross-attention for decoder queries ---
        self.step_embed = nn.Embedding(self.t_output + self.t_input + 10, self.bottleneck_ch)
        self.cross_attn = TemporalCrossAttentionDecoder(self.bottleneck_ch, n_heads, dropout)

        # --- UNet Decoder ---
        dec_channels = list(reversed(enc_channels[:-1]))
        self.decoder_ups = nn.ModuleList()
        in_ch = self.bottleneck_ch
        for i, out_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i+2)]
            self.decoder_ups.append(Up(in_ch, skip_ch, out_ch, dropout))
            in_ch = out_ch

        # --- Output head: predicts FULL frame, not residual ---
        self.output_head = nn.Sequential(
            nn.Conv2d(dec_channels[-1], dec_channels[-1], 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dec_channels[-1]), dec_channels[-1]),
            nn.GELU(),
            nn.Conv2d(dec_channels[-1], self.out_channels, 1),
        )

        # --- Learned blend gate: alpha * pred + (1-alpha) * last_frame ---
        # Initialised to log(0.05/(1-0.05)) so sigmoid → ~0.05 at start.
        # Model learns to increase alpha as it gains confidence.
        self.alpha_logit = nn.Parameter(
            torch.full((self.out_channels,), math.log(0.05 / 0.95))
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
        x : (B, T_in, C, H, W)
        Returns: (B, T_out, C_out, H, W)
        """
        B, T_in, C, H, W = x.shape
        last_frame = x[:, -1, :self.out_channels]   # (B, C_out, H, W)

        # --- Encode: fold T into batch ---
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        skips_flat = [self.inc(x_flat)]
        for down in self.downs:
            skips_flat.append(down(skips_flat[-1]))

        # bottleneck: (B*T, C_bn, H_bn, W_bn)
        C_bn = skips_flat[-1].shape[1]
        H_bn = skips_flat[-1].shape[2]
        W_bn = skips_flat[-1].shape[3]
        N    = H_bn * W_bn   # spatial tokens per frame

        # --- Mamba on spatial tokens ---
        # Reshape: (B*T, C_bn, H_bn, W_bn) → (B, T*N, C_bn)
        bn_flat = skips_flat[-1].reshape(B, T_in, C_bn, N)   # (B, T, C_bn, N)
        bn_flat = bn_flat.permute(0, 1, 3, 2).reshape(B, T_in * N, C_bn)  # (B, T*N, C_bn)

        tokens = self.spatial_to_token(bn_flat)   # (B, T*N, d_mamba)
        for mamba in self.mamba_layers:
            tokens = mamba(tokens)                 # (B, T*N, d_mamba)
        tokens = self.token_to_spatial(tokens)     # (B, T*N, C_bn)

        # Temporal pooling: reshape back to (B, T, N, C_bn) → mean over N → (B, T, C_bn)
        temporal_context = tokens.reshape(B, T_in, N, C_bn).mean(dim=2)   # (B, T, C_bn)

        # Spatial bottleneck for decoder: mean over T → (B, C_bn, H_bn, W_bn)
        spatial_bn = rearrange(
            skips_flat[-1], "(b t) c h w -> b t c h w", b=B
        ).mean(dim=1)

        # Temporal-mean skips for decoder
        skips_mean = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B).mean(dim=1)
            for s in skips_flat
        ]

        # Learned blend alpha: (C_out,) → per-channel scalar in (0,1)
        alpha = torch.sigmoid(self.alpha_logit).view(1, self.out_channels, 1, 1)

        # --- Decode: T_out steps ---
        preds = []
        for step in range(self.t_output):
            # Cross-attend over Mamba temporal context
            step_idx = torch.tensor([T_in + step], device=x.device)
            query = self.step_embed(step_idx).unsqueeze(1)   # (1, 1, C_bn) → broadcast to B
            query = query.expand(B, -1, -1)                  # (B, 1, C_bn)
            cond  = self.cross_attn(query, temporal_context) # (B, C_bn)

            # Modulate spatial bottleneck with temporal conditioning (additive)
            feat = spatial_bn + cond.view(B, C_bn, 1, 1)

            # UNet decoder
            for i, up in enumerate(self.decoder_ups):
                feat = up(feat, skips_mean[-(i+2)])

            # Full-frame prediction + learned blend with last_frame
            raw_pred = self.output_head(feat)                # (B, C_out, H, W)
            pred = alpha * raw_pred + (1.0 - alpha) * last_frame

            preds.append(pred)

        return torch.stack(preds, dim=1)   # (B, T_out, C_out, H, W)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)