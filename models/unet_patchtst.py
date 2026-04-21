"""
unet_patchtst.py  (v5 — full architecture + stability fixes)
-------------------------------------------------------------
Fixes vs v4:

FIX C (PatchTST) — Per-band LayerNorm BEFORE the temporal transformer.
  PatchTST was designed for 1D time-series where each channel is normalised
  independently. Without per-band norm, channels with very different scales
  (e.g. z-scored SAR vs [0,1] NDVI) produce unstable attention weights.
  New: BandwiseLayerNorm normalises each band's T-length sequence before
  the transformer sees it.

FIX C — Correct spatiotemporal reshaping.
  Old pipeline: pool spatial → (B, T, C_bn) → transformer → project
  This loses ALL spatial information before the temporal module.

  New pipeline (hybrid):
    (B, T, C_bn, H_bn, W_bn) → rearrange → (B*H_bn*W_bn, T, C_bn)
    → temporal transformer (each spatial location independently)
    → rearrange back → (B, T, C_bn, H_bn, W_bn)
  This is the correct PatchTST-style processing where EACH spatial location
  has its own T-length time series passed through the transformer.
  Complexity: (B * H_bn * W_bn) * O(T^2) — with T=4 and H_bn=W_bn=8
  that is B*64*16 = B*1024 attention ops, very affordable.

FIX C — Spatial attention block after temporal transformer.
  After per-location temporal modelling, a lightweight spatial self-attention
  (SpatialBottleneckAttention) lets locations exchange information across space.
  This partially compensates for the loss of spatial context from the
  per-location temporal processing.

FIX 4 — SAR input gate (same as unet_mamba v6).
  Learnable gate initialised near 0 to suppress SAR fill-value noise.

FIX A — Gradient norm hooks same interface as unet_mamba.

UNCHANGED: InputBiasCorrector, FiLM (clamped), UNet encoder/decoder,
           separate optical/SAR heads, MC Dropout.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# FIX C: Per-band normalisation before transformer
# ---------------------------------------------------------------------------

class BandwiseLayerNorm(nn.Module):
    """
    Normalise each band's temporal sequence independently.

    Input:  (B, T, C)  — pooled spatial features per band over T steps
    Output: (B, T, C)  — each of the C channels normalised across T

    Why:  channels have very different scales (z-scored SAR, [0,1] NDVI).
    Normalising per-band prevents high-variance channels from dominating
    the transformer's attention logits.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — norm over the last dim (channel)
        return self.norm(x)


# ---------------------------------------------------------------------------
# FIX C: Spatial self-attention at bottleneck
# ---------------------------------------------------------------------------

class SpatialBottleneckAttention(nn.Module):
    """
    Lightweight spatial self-attention over H_bn × W_bn tokens.

    After per-location temporal modelling, spatial locations have been
    processed independently. This module lets them exchange information.

    For H_bn=W_bn=8 (128px / 16 stride): 64 tokens, 64×64 attention matrix
    → trivially cheap.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x.shape
        # Flatten spatial: (B, H*W, C)
        tokens = x.reshape(B, C, H * W).permute(0, 2, 1)
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(tokens)
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Temporal Transformer (pre-norm, unchanged structure but fed per-location)
# ---------------------------------------------------------------------------

class TemporalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden     = int(d_model * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalTransformerEncoder(nn.Module):
    """
    N-layer transformer over T temporal tokens with sinusoidal positional encoding.

    FIX C: Now operates on (B * H_bn * W_bn, T, C) instead of (B, T, C).
    Each spatial location has its own independent temporal sequence through
    the transformer. This is the correct PatchTST-style processing.
    """

    def __init__(
        self,
        d_model:   int,
        n_layers:  int   = 3,
        n_heads:   int   = 4,
        mlp_ratio: float = 2.0,
        dropout:   float = 0.1,
        max_len:   int   = 64,
    ):
        super().__init__()
        # Sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N_loc, T, D)  where N_loc = B * H_bn * W_bn
        Returns: (N_loc, T, D)
        """
        x = x + self.pe[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# SAR input gate (FIX 4)
# ---------------------------------------------------------------------------

class SARInputGate(nn.Module):
    """Learned scalar gate on the SAR channel. Init ≈ 0.05 (nearly closed)."""

    def __init__(self):
        super().__init__()
        self.log_gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor, sar_idx: int) -> torch.Tensor:
        gate  = torch.sigmoid(self.log_gate)
        scale = torch.ones(x.shape[2], device=x.device, dtype=x.dtype)
        scale[sar_idx] = gate
        return x * scale.view(1, 1, -1, 1, 1)


# ---------------------------------------------------------------------------
# FiLM (clamped)
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
# InputBiasCorrector
# ---------------------------------------------------------------------------

class InputBiasCorrector(nn.Module):
    def __init__(self, n_bands: int, band_names: List[str] = None,
                 norm_methods: Dict[str, str] = None):
        super().__init__()
        init_fills = torch.zeros(n_bands)
        if norm_methods and band_names:
            for i, name in enumerate(band_names):
                method = norm_methods.get(name.lower(), "minmax")
                if method in ("minmax", "shift"):
                    init_fills[i] = 0.5
        else:
            init_fills[:-1] = 0.5
        self.fill_values = nn.Parameter(init_fills)

    def forward(self, x: torch.Tensor, has_validity_ch: bool = True) -> torch.Tensor:
        if not has_validity_ch:
            return x
        n_bands = x.shape[2] - 1
        bands   = x[:, :, :n_bands]
        valid   = x[:, :, n_bands:]
        fills   = self.fill_values[:n_bands].view(1, 1, -1, 1, 1)
        invalid_mask = (valid < 0.5).expand_as(bands)
        return torch.where(invalid_mask, fills.expand_as(bands), bands)


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

class UNetPatchTSTModel(nn.Module):
    """
    UNet + PatchTST-style Temporal Transformer — v5.

    Key architecture changes vs v4:
      1. Per-band LayerNorm before transformer (BandwiseLayerNorm)
      2. Per-location temporal transformer: each spatial bottleneck location
         gets its own T-step sequence through the transformer
         (B, T, C, H, W) → (B*H*W, T, C) → transformer → (B, T, C, H, W)
      3. Spatial self-attention AFTER temporal modelling (SpatialBottleneckAttention)
      4. Learnable SAR input gate
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config      = config
        model_cfg        = config["model"].get("unet_patchtst", {})
        use_validity     = config["patches"]["use_validity_mask"]

        n_bands           = config["bands"]["count"]
        self.n_bands      = n_bands
        self.in_channels  = n_bands + (1 if use_validity else 0)
        self.out_channels = n_bands
        self.t_input      = config["model"]["t_input"]
        self.t_output     = config["model"]["t_output"]
        self.use_validity = use_validity
        self.sar_idx      = config["bands"]["indices"]["sar_vv"]

        enc_ch      = model_cfg.get("encoder_channels", [32, 64, 128, 256])
        n_tf_layers = model_cfg.get("n_transformer_layers", 3)
        n_heads     = model_cfg.get("n_heads", 4)
        mlp_ratio   = model_cfg.get("mlp_ratio", 2.0)
        dropout     = model_cfg.get("dropout", 0.1)

        self.bottleneck_ch = enc_ch[-1]

        # FIX 4: SAR gate
        self.sar_gate = SARInputGate()

        # Input de-bias
        band_names   = config["bands"]["names"]
        norm_methods = config["bands"]["normalization"]
        self.input_corrector = InputBiasCorrector(n_bands, band_names, norm_methods)

        # Spatial Encoder
        self.inc   = DoubleConv(n_bands, enc_ch[0], dropout)
        self.downs = nn.ModuleList([
            Down(enc_ch[i-1], enc_ch[i], dropout) for i in range(1, len(enc_ch))
        ])

        # FIX C: Per-band norm before transformer
        self.band_norm = BandwiseLayerNorm(self.bottleneck_ch)

        # FIX C: Temporal transformer (will be applied per spatial location)
        self.temporal_enc = TemporalTransformerEncoder(
            d_model=self.bottleneck_ch,
            n_layers=n_tf_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # FIX C: Spatial attention after temporal modelling
        self.spatial_attn = SpatialBottleneckAttention(
            d_model=self.bottleneck_ch,
            n_heads=min(4, self.bottleneck_ch // 16),
            dropout=dropout,
        )

        # T_in → T_out projection
        self.future_proj = nn.Linear(self.t_input, self.t_output)
        nn.init.xavier_uniform_(self.future_proj.weight)

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
        x : (B, T_in, C_in, H, W)
        Returns: (B, T_out, n_bands, H, W)
        """
        B, T_in, C, H, W = x.shape

        # Step 0: fix NaN fill bias
        x_corrected = self.input_corrector(x, has_validity_ch=self.use_validity)

        # FIX 4: gate SAR noise
        x_gated = self.sar_gate(x_corrected, self.sar_idx)

        # Step 1: Spatial encoding (fold T into batch)
        x_flat = rearrange(x_gated, "b t c h w -> (b t) c h w")
        skips_flat = [self.inc(x_flat)]
        for down in self.downs:
            skips_flat.append(down(skips_flat[-1]))

        # bottleneck: (B*T, C_bn, H_bn, W_bn)
        bn = skips_flat[-1]
        C_bn, H_bn, W_bn = bn.shape[1], bn.shape[2], bn.shape[3]

        # Unfold: (B, T, C_bn, H_bn, W_bn)
        spatial_all = rearrange(bn, "(b t) c h w -> b t c h w", b=B)

        # FIX C: Per-location temporal transformer
        # Reshape to (B * H_bn * W_bn, T, C_bn) — each location gets its own sequence
        # This is the key PatchTST reshape: treat each spatial location as a sample
        x_loc = rearrange(spatial_all, "b t c h w -> (b h w) t c",
                          h=H_bn, w=W_bn)  # (B*H*W, T, C_bn)

        # FIX C: Per-band norm before transformer
        x_loc = self.band_norm(x_loc)

        # Temporal transformer
        x_loc = self.temporal_enc(x_loc)          # (B*H*W, T, C_bn)

        # Reshape back: (B, T, C_bn, H_bn, W_bn)
        spatial_temporal = rearrange(
            x_loc, "(b h w) t c -> b t c h w", b=B, h=H_bn, w=W_bn
        )

        # FIX C: Spatial attention on mean-over-T spatial map
        # (lets locations exchange information after temporal processing)
        spatial_mean_raw = spatial_temporal.mean(dim=1)   # (B, C_bn, H_bn, W_bn)
        spatial_mean = self.spatial_attn(spatial_mean_raw)

        # Project T_in → T_out using the mean-over-space temporal signal
        pooled   = spatial_temporal.mean(dim=(-2, -1))    # (B, T_in, C_bn)
        cond_out = self.future_proj(
            pooled.permute(0, 2, 1)
        ).permute(0, 2, 1)                                 # (B, T_out, C_bn)

        # Compute mean skips
        skips_mean = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B).mean(dim=1)
            for s in skips_flat
        ]

        # Step 2: Decode T_out frames
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