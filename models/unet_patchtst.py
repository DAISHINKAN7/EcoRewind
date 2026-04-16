"""
unet_patchtst.py  (v3 — correct architecture)
----------------------------------------------
Same correct pipeline as unet_mamba but with a Transformer
temporal module instead of Mamba.

For T_in=4, we use direct self-attention (4×4 matrix) — no patching.
Patching a 4-element sequence reduces it to 2 tokens which loses
temporal resolution without any computational benefit at this scale.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Temporal Transformer
# ---------------------------------------------------------------------------

class TemporalTransformerBlock(nn.Module):
    """Pre-norm transformer block. Input/Output: (B, T, D)"""

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
        x = x + self.drop(self.attn(normed, normed, normed)[0])
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalTransformerEncoder(nn.Module):
    """
    N-layer transformer over T temporal tokens.
    Includes sinusoidal positional encoding.
    Input/Output: (B, T, D)
    """

    def __init__(self, d_model: int, n_layers: int = 3, n_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, D)

        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, D) → (B, T, D)"""
        x = x + self.pe[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# FiLM fusion
# ---------------------------------------------------------------------------

class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation.
    Initialised near identity for stable training start.
    """

    def __init__(self, d_cond: int, d_spatial: int):
        super().__init__()
        self.proj = nn.Linear(d_cond, 2 * d_spatial)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
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

class UNetPatchTSTModel(nn.Module):
    """
    UNet + Temporal Transformer spatiotemporal model.

    Identical pipeline to UNetMambaModel, transformer replaces Mamba:
      Encode T_in → pool → Transformer over T → project T_in→T_out
      → FiLM modulate spatial bottleneck per output step → decode
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

        enc_ch      = model_cfg.get("encoder_channels", [32, 64, 128, 256])
        n_tf_layers = model_cfg.get("n_transformer_layers", 3)
        n_heads     = model_cfg.get("n_heads", 4)
        mlp_ratio   = model_cfg.get("mlp_ratio", 2.0)
        dropout     = model_cfg.get("dropout", 0.1)

        self.bottleneck_ch = enc_ch[-1]

        # ---- Step 1: Spatial Encoder ----
        self.inc   = DoubleConv(self.in_channels, enc_ch[0], dropout)
        self.downs = nn.ModuleList([
            Down(enc_ch[i-1], enc_ch[i], dropout)
            for i in range(1, len(enc_ch))
        ])

        # ---- Step 2: Temporal Transformer ----
        self.temporal_enc = TemporalTransformerEncoder(
            d_model=self.bottleneck_ch,
            n_layers=n_tf_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Project T_in encoder outputs → T_out conditioning vectors
        self.future_proj = nn.Linear(self.t_input, self.t_output)

        # ---- Step 3: FiLM fusion ----
        self.film_bn = FiLMFusion(self.bottleneck_ch, self.bottleneck_ch)

        # ---- Step 4: Decoder ----
        dec_ch = list(reversed(enc_ch[:-1]))
        self.decoder_ups = nn.ModuleList()
        in_ch = self.bottleneck_ch
        for i, out_ch_d in enumerate(dec_ch):
            skip_ch = enc_ch[-(i+2)]
            self.decoder_ups.append(Up(in_ch, skip_ch, out_ch_d, dropout))
            in_ch = out_ch_d

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

        # ================================================================
        # Step 1: Spatial encoding — all T frames in parallel
        # ================================================================
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")

        skips_flat = [self.inc(x_flat)]
        for down in self.downs:
            skips_flat.append(down(skips_flat[-1]))
        # skips_flat[-1]: (B*T, C_bn, H_bn, W_bn)

        C_bn = skips_flat[-1].shape[1]

        # Unfold: (B, T, C_bn, H_bn, W_bn)
        spatial_all = rearrange(skips_flat[-1], "(b t) c h w -> b t c h w", b=B)

        # ================================================================
        # Step 2: Temporal modeling
        # Pool spatial dims → (B, T, C_bn) → Transformer → (B, T, C_bn)
        # ================================================================
        pooled = spatial_all.mean(dim=(-2, -1))              # (B, T_in, C_bn)
        temporal = self.temporal_enc(pooled)                 # (B, T_in, C_bn)

        # Project T_in → T_out conditioning vectors
        # (B, T_in, C_bn) → perm → (B, C_bn, T_in) → linear → (B, C_bn, T_out)
        # → perm → (B, T_out, C_bn)
        cond_out = self.future_proj(
            temporal.permute(0, 2, 1)
        ).permute(0, 2, 1)                                   # (B, T_out, C_bn)

        # ================================================================
        # Step 3: Spatial base + skips
        # ================================================================
        spatial_mean = spatial_all.mean(dim=1)               # (B, C_bn, H_bn, W_bn)

        skips_mean = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B).mean(dim=1)
            for s in skips_flat
        ]

        # ================================================================
        # Step 4: Decode T_out frames with distinct temporal conditioning
        # ================================================================
        preds = []
        for step in range(self.t_output):
            cond_vec = cond_out[:, step, :]                  # (B, C_bn)

            # FiLM: spatial map gets DIFFERENT modulation per output step
            feat = self.film_bn(spatial_mean, cond_vec)      # (B, C_bn, H_bn, W_bn)

            for i, up in enumerate(self.decoder_ups):
                feat = up(feat, skips_mean[-(i+2)])

            pred = self.output_head(feat)                    # (B, C_out, H, W)
            preds.append(pred)

        return torch.stack(preds, dim=1)                     # (B, T_out, C_out, H, W)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)