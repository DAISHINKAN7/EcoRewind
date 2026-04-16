"""
unet_patchtst.py  (v2 — fixed lazy prediction)
-----------------------------------------------
Same root cause fix as unet_mamba v2:
  1. Removes delta_scale bottleneck.
  2. Output head predicts full frame, blended with last_frame via
     learned per-channel alpha gate.
  3. PatchTST transformer operates on the full sequence of spatial
     bottleneck pooled features (per-timestep, not patched aggregate).
     Patching a T=4 sequence into K=2 patches then averaging collapses
     temporal information too aggressively for such a short sequence.
     Instead we use direct self-attention over T=4 tokens (16-element
     attention matrix — essentially free) which gives full temporal
     resolution.
  4. Decoder uses cross-attention over all T_in transformer outputs
     instead of just the last one.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Lightweight temporal transformer (direct attention, no patching for T<=8)
# ---------------------------------------------------------------------------

class TemporalSelfAttention(nn.Module):
    """
    Standard pre-norm transformer block over temporal sequence.
    For T=4 this is a 4×4 attention matrix — negligible cost.
    Patching a 4-element sequence loses temporal resolution unnecessarily.
    """
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
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalTransformerEncoder(nn.Module):
    """
    N-layer transformer over the T_in temporal sequence of bottleneck
    pooled features.  Produces one output vector per timestep.
    """
    def __init__(self, d_model: int, n_layers: int = 3, n_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        # Sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
        self.register_buffer("pe", pe.unsqueeze(0))

        self.blocks = nn.ModuleList([
            TemporalSelfAttention(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, D) → (B, T, D)"""
        x = x + self.pe[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class TemporalCrossAttentionDecoder(nn.Module):
    """Each decode step attends over all T encoder outputs."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """query: (B,1,D), keys: (B,T,D) → (B,D)"""
        q = self.norm_q(query)
        k = self.norm_k(keys)
        out, _ = self.attn(q, k, k)
        out = query + self.drop(out)
        return self.norm(out).squeeze(1)


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

class UNetPatchTSTModel(nn.Module):
    """
    UNet + Temporal Transformer for satellite time-series forecasting.

    Changes from v1:
    - Direct self-attention over T=4 tokens (not PatchTST patching).
      Patching T=4 into K=2 patches then averaging gives 2 tokens — that's
      just a linear projection followed by averaging, not temporal attention.
    - Learned alpha blend gate replaces fixed delta_scale.
    - Cross-attention decoder instead of last-hidden-state.
    - No output clamping.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config      = config
        model_cfg        = config["model"].get("unet_patchtst", {})
        use_validity     = config["patches"]["use_validity_mask"]

        self.in_channels  = config["bands"]["count"] + (1 if use_validity else 0)
        self.out_channels = config["bands"]["count"]
        self.t_input  = config["model"]["t_input"]
        self.t_output = config["model"]["t_output"]

        enc_channels  = model_cfg.get("encoder_channels", [32, 64, 128, 256])
        n_tf_layers   = model_cfg.get("n_transformer_layers", 3)
        n_heads       = model_cfg.get("n_heads", 4)
        mlp_ratio     = model_cfg.get("mlp_ratio", 2.0)
        dropout       = model_cfg.get("dropout", 0.1)

        self.bottleneck_ch = enc_channels[-1]

        # --- UNet Encoder ---
        self.inc  = DoubleConv(self.in_channels, enc_channels[0], dropout)
        self.downs = nn.ModuleList([
            Down(enc_channels[i-1], enc_channels[i], dropout)
            for i in range(1, len(enc_channels))
        ])

        # --- Temporal transformer encoder ---
        self.temporal_enc = TemporalTransformerEncoder(
            d_model=self.bottleneck_ch,
            n_layers=n_tf_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # --- Temporal cross-attention for decoder ---
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

        # --- Full-frame output head + learned blend gate ---
        self.output_head = nn.Sequential(
            nn.Conv2d(dec_channels[-1], dec_channels[-1], 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dec_channels[-1]), dec_channels[-1]),
            nn.GELU(),
            nn.Conv2d(dec_channels[-1], self.out_channels, 1),
        )

        # Learned blend: starts at alpha≈0.05, grows as model learns
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
        last_frame = x[:, -1, :self.out_channels]

        # --- UNet encode ---
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        skips_flat = [self.inc(x_flat)]
        for down in self.downs:
            skips_flat.append(down(skips_flat[-1]))

        C_bn = skips_flat[-1].shape[1]

        # --- Temporal transformer on spatially pooled bottleneck ---
        pooled = skips_flat[-1].mean(dim=(-2, -1))                    # (B*T, C_bn)
        temporal_seq = rearrange(pooled, "(b t) c -> b t c", b=B)     # (B, T, C_bn)
        temporal_ctx = self.temporal_enc(temporal_seq)                 # (B, T, C_bn)

        # Spatial bottleneck mean over T → (B, C_bn, H_bn, W_bn)
        spatial_bn = rearrange(
            skips_flat[-1], "(b t) c h w -> b t c h w", b=B
        ).mean(dim=1)

        # Temporal-mean skips
        skips_mean = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B).mean(dim=1)
            for s in skips_flat
        ]

        alpha = torch.sigmoid(self.alpha_logit).view(1, self.out_channels, 1, 1)

        # --- Decode ---
        preds = []
        for step in range(self.t_output):
            step_idx = torch.tensor([T_in + step], device=x.device)
            query = self.step_embed(step_idx).unsqueeze(1).expand(B, -1, -1)  # (B,1,D)
            cond  = self.cross_attn(query, temporal_ctx)                        # (B, D)

            feat = spatial_bn + cond.view(B, C_bn, 1, 1)

            for i, up in enumerate(self.decoder_ups):
                feat = up(feat, skips_mean[-(i+2)])

            raw_pred = self.output_head(feat)
            pred = alpha * raw_pred + (1.0 - alpha) * last_frame
            preds.append(pred)

        return torch.stack(preds, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)