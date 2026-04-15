"""
unet_patchtst.py
----------------
UNet + Patch-based Temporal Transformer (PatchTST-style) for ECO-REWIND.

Architecture:
  1. UNet encoder processes each frame independently.
  2. PatchTST-style temporal module:
     - Spatial features are pooled into a compact representation.
     - Temporal dimension is divided into PATCHES (groups of K timesteps).
     - Each patch is embedded as a single token.
     - A lightweight transformer operates only over the patch tokens (NOT pixels).
     - This gives O(T/K)² complexity instead of O(T²) for full attention.
  3. UNet decoder reconstructs future frames using FiLM conditioning.

Why PatchTST-style over standard ViT:
  - Standard ViT applied to (T, H, W) input has O((T·H·W)²) attention — prohibitive.
  - PatchTST insight: patch along the TIME axis, not spatial axis.
  - Patching T into K-timestep groups reduces sequence length from T to T/K.
  - Keeps all spatial structure intact (no pixel patching).
  - Suitable for our case: T=4–32 quarters, K=2 → 2–16 patch tokens.

~7M parameters at default config.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Patch-based temporal embedding
# ---------------------------------------------------------------------------

class TemporalPatchEmbedding(nn.Module):
    """
    Divide temporal sequence into patches and embed each.

    Input:  (B, T, d_model)   — temporal feature sequence
    Output: (B, n_patches, d_model)

    Each patch covers K consecutive timesteps.
    If T is not divisible by K, the sequence is zero-padded.

    This is the core PatchTST insight: treating K consecutive timesteps
    as a single "temporal token" dramatically reduces sequence length.
    """

    def __init__(self, d_model: int, patch_len: int = 2, stride: int = 1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        # Linear projection: K × d_model → d_model
        self.proj = nn.Linear(patch_len * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, D)
        Returns: (B, n_patches, D)
        """
        B, T, D = x.shape
        K = self.patch_len
        s = self.stride

        # Pad T so it's divisible by stride
        n_patches = max(1, (T - K) // s + 1)
        patches = []
        for i in range(n_patches):
            start = i * s
            end   = start + K
            if end > T:
                # Pad last patch
                patch = x[:, start:T]
                pad_len = end - T
                patch = F.pad(patch, (0, 0, 0, pad_len))
            else:
                patch = x[:, start:end]
            patches.append(patch.reshape(B, K * D))   # (B, K*D)

        patches = torch.stack(patches, dim=1)          # (B, n_patches, K*D)
        return self.norm(self.proj(patches))            # (B, n_patches, D)


class SinusoidalPosEncoding1D(nn.Module):
    """Sinusoidal 1-D positional encoding for temporal patches."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, D)"""
        return x + self.pe[:, :x.shape[1]]


# ---------------------------------------------------------------------------
# Lightweight transformer encoder (temporal only)
# ---------------------------------------------------------------------------

class TemporalTransformerBlock(nn.Module):
    """
    Single transformer block operating over temporal patch tokens.

    Pre-LayerNorm design for stability.
    Uses flash-attention compatible multi-head attention.
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
        # Self-attention over patch tokens
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class PatchTemporalTransformer(nn.Module):
    """
    Full PatchTST-style temporal encoder.

    Pipeline:
      (B, T, d_model)
        → TemporalPatchEmbedding → (B, n_patches, d_model)
        → SinusoidalPosEncoding
        → N × TemporalTransformerBlock
        → (B, n_patches, d_model)
        → Linear unpatching → (B, T, d_model)

    The unpatching step reconstructs a per-timestep representation from
    patch tokens using a learned linear projection. This gives one output
    vector per original timestep, which feeds into the FiLM decoder.
    """

    def __init__(self, d_model: int, patch_len: int = 2, stride: int = 1,
                 n_layers: int = 2, n_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, t_input: int = 4):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.t_input   = t_input

        self.patch_embed = TemporalPatchEmbedding(d_model, patch_len, stride)
        self.pos_enc     = SinusoidalPosEncoding1D(d_model, max_len=64)

        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Compute n_patches for the fixed t_input
        n_patches = max(1, (t_input - patch_len) // stride + 1)
        # Unpatch: map n_patches tokens back to t_input timestep vectors
        # Simple: tile + linear project
        self.unpatch = nn.Linear(d_model, d_model * t_input, bias=False)
        self.unpatch_norm = nn.LayerNorm(d_model)
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d_model)
        Returns: (B, T, d_model)  — per-timestep features after temporal attention
        """
        B, T, D = x.shape
        residual = x

        # Patch embedding: (B, T, D) → (B, n_patches, D)
        patches = self.patch_embed(x)
        patches = self.pos_enc(patches)

        # Transformer over patches
        for block in self.blocks:
            patches = block(patches)
        patches = self.norm(patches)                    # (B, n_patches, D)

        # Unpatch: aggregate patch tokens back to T timestep vectors
        # Average all patch tokens → single context vector → project to T*D
        ctx = patches.mean(dim=1)                       # (B, D)
        unpacked = self.unpatch(ctx).view(B, T, D)      # (B, T, D)
        unpacked = self.unpatch_norm(unpacked)

        # Residual: blend original temporal features with transformer output
        return residual + unpacked


# ---------------------------------------------------------------------------
# Shared building blocks
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


class FiLMLayer(nn.Module):
    def __init__(self, d_cond: int, d_spatial: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_cond, d_cond), nn.GELU(),
            nn.Linear(d_cond, 2 * d_spatial),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta  = beta.view(-1, x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UNetPatchTSTModel(nn.Module):
    """
    UNet + PatchTST-style temporal transformer for satellite time-series.

    Key difference from UNetMamba:
    - Uses attention-based temporal modeling (better for irregular patterns,
      seasonal cycles, multi-scale dependencies).
    - PatchTST patching reduces sequence length → memory-efficient attention.
    - Better when temporal patterns are periodic (seasons) or non-Markovian.

    Use UNetMamba when:
    - Very long sequences (T > 20 quarters) where O(T²) is prohibitive.
    - Streaming/online inference needed (Mamba is recurrent at inference).

    Use UNetPatchTST when:
    - Moderate T (4–20 quarters), seasonal patterns matter.
    - Best offline batch training performance is the goal.
    - Interpretability matters (attention weights show which timesteps matter).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_cfg   = config["model"].get("unet_patchtst", {})
        use_validity = config["patches"]["use_validity_mask"]

        self.in_channels  = config["bands"]["count"] + (1 if use_validity else 0)
        self.out_channels = config["bands"]["count"]
        self.t_input  = config["model"]["t_input"]
        self.t_output = config["model"]["t_output"]

        enc_channels  = model_cfg.get("encoder_channels", [32, 64, 128, 256])
        patch_len     = model_cfg.get("patch_len", 2)
        stride        = model_cfg.get("stride", 1)
        n_tf_layers   = model_cfg.get("n_transformer_layers", 2)
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

        # --- PatchTST temporal module ---
        self.temporal_tf = PatchTemporalTransformer(
            d_model=self.bottleneck_ch,
            patch_len=patch_len,
            stride=stride,
            n_layers=n_tf_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            t_input=self.t_input,
        )

        # --- FiLM conditioning ---
        self.film = FiLMLayer(self.bottleneck_ch, self.bottleneck_ch)

        # --- UNet Decoder ---
        dec_channels = list(reversed(enc_channels[:-1]))
        self.decoder_ups = nn.ModuleList()
        in_ch = self.bottleneck_ch
        for i, out_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i+2)]
            self.decoder_ups.append(Up(in_ch, skip_ch, out_ch, dropout))
            in_ch = out_ch

        # --- Residual output head ---
        self.output_head = nn.Sequential(
            nn.Conv2d(dec_channels[-1], dec_channels[-1] // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dec_channels[-1] // 2), dec_channels[-1] // 2),
            nn.GELU(),
            nn.Conv2d(dec_channels[-1] // 2, self.out_channels, 1),
            nn.Tanh(),
        )
        self.delta_scale = nn.Parameter(torch.zeros(1))
        self.pos_embed   = nn.Embedding(self.t_output + self.t_input + 10, self.bottleneck_ch)
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

        # --- Encode ---
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        skips_flat = [self.inc(x_flat)]
        for down in self.downs:
            skips_flat.append(down(skips_flat[-1]))

        C_bn = skips_flat[-1].shape[1]

        # --- PatchTST temporal module ---
        pooled = skips_flat[-1].mean(dim=(-2, -1))                    # (B*T, C_bn)
        temporal_seq = rearrange(pooled, "(b t) c -> b t c", b=B)     # (B, T, C_bn)
        temporal_seq = self.temporal_tf(temporal_seq)                  # (B, T, C_bn)

        # Spatial bottleneck (mean over T)
        spatial_bn = rearrange(
            skips_flat[-1], "(b t) c h w -> b t c h w", b=B
        ).mean(dim=1)                                                  # (B, C_bn, H_bn, W_bn)

        # Temporal-mean skips for decoder
        skips_mean = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B).mean(dim=1)
            for s in skips_flat
        ]

        # --- Decode ---
        preds = []
        dec_state = temporal_seq[:, -1]   # (B, C_bn)

        for step in range(self.t_output):
            pos_emb = self.pos_embed(
                torch.tensor([T_in + step], device=x.device)
            ).squeeze(0)
            query = dec_state + pos_emb.unsqueeze(0)

            feat = self.film(spatial_bn, query)

            for i, up in enumerate(self.decoder_ups):
                feat = up(feat, skips_mean[-(i+2)])

            delta = self.output_head(feat)
            scale = torch.sigmoid(self.delta_scale) * 0.5
            pred  = last_frame + delta * scale
            preds.append(pred)

            dec_state = temporal_seq[:, min(step, T_in - 1)]

        return torch.stack(preds, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)