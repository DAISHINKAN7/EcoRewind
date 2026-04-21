"""
eco_transformer.py  (v3 — fp16-safe output clamping + tighter gate init)
-------------------------------------------------------------------------
Changes vs v2:

FIX T1 — Replace in-place preds_clamped[:, :, c] = ... with torch.stack.
  The v2 forward created `preds_clamped = preds.clone()` then did per-band
  in-place assignment inside a Python loop. This generates a CopySlices
  autograd node for each band (confirmed in the v6 diagnostic:
  `pred.grad_fn : <CopySlices object>`). CopySlices is valid but creates
  extra autograd overhead and can cause inefficient backward passes.
  Using torch.stack along the band dim is cleaner and faster.

FIX T2 — Tighter output clamp.
  v2 clamped optical bands to [-0.1, 1.1] and SAR to [-6, 6]. With the
  normalised target ranges (optical in [0,1], z-scored SAR with std=1),
  the SAR clamp of ±6 is 6σ — allowing massive residuals. Tightened to
  [-4, 4] for SAR, matching the ecological-loss bounds and ±4σ coverage.
  For optical, kept [-0.1, 1.1] which is a light safety margin.

UNCHANGED: SAR input gate, ChannelImportanceWeighting, SSIM decoder,
           factorized ST blocks, cross-attention query tokens, MC dropout.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# SAR input gate
# ---------------------------------------------------------------------------

class SARInputGate(nn.Module):
    def __init__(self, sar_idx: int, n_channels: int):
        super().__init__()
        self.sar_idx  = sar_idx
        self.n_ch     = n_channels
        self.log_gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate  = torch.sigmoid(self.log_gate)
        scale = torch.ones(self.n_ch, device=x.device, dtype=x.dtype)
        scale[self.sar_idx] = gate
        return x * scale.view(1, -1, 1, 1)


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

class SinusoidalPos2D(nn.Module):
    def __init__(self, d_model: int, h: int, w: int):
        super().__init__()
        pe = torch.zeros(h * w, d_model)
        positions = torch.arange(h * w, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class LearnedTemporalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, t: int, device: torch.device) -> torch.Tensor:
        idx = torch.tensor([t], device=device)
        return self.embedding(idx).unsqueeze(0)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class PreNormAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask=None):
        q_norm = self.norm(q)
        attended, _ = self.attn(q_norm, k, v, key_padding_mask=key_padding_mask)
        return q + self.drop(attended)


class PreNormFFN(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class FactorizedSTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.temporal_attn = PreNormAttention(d_model, n_heads, dropout)
        self.spatial_attn  = PreNormAttention(d_model, n_heads, dropout)
        self.ffn = PreNormFFN(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x_t = self.temporal_attn(x_t, x_t, x_t)
        x = x_t.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x_s = x.reshape(B * T, N, D)
        x_s = self.spatial_attn(x_s, x_s, x_s)
        x = x_s.reshape(B, T, N, D)
        x = self.ffn(x.reshape(B * T, N, D)).reshape(B, T, N, D)
        return x


class ChannelImportanceWeighting(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.log_w = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.exp(self.log_w).view(1, -1, 1, 1)
        return x * w


# ---------------------------------------------------------------------------
# Encoder/decoder
# ---------------------------------------------------------------------------

class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int, sar_idx: int = 6):
        super().__init__()
        mid = d_model // 2
        self.sar_gate = SARInputGate(sar_idx, in_channels)
        self.ch_weight = ChannelImportanceWeighting(in_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.GELU(),
            nn.Conv2d(mid // 2, mid, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, d_model, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model), nn.GELU(),
            nn.Conv2d(d_model, d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sar_gate(x)
        x = self.ch_weight(x)
        return self.encoder(x)


class SpatialDecoder(nn.Module):
    def __init__(self, d_model: int, out_channels: int):
        super().__init__()
        mid = d_model // 2
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d_model, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid // 2, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


# ---------------------------------------------------------------------------
# EcoTransformer
# ---------------------------------------------------------------------------

class EcoTransformer(nn.Module):
    """
    Factorized Spatiotemporal Transformer — v3.

    Changes vs v2:
      - Output clamping uses torch.stack instead of in-place slice assignment
        (avoids CopySlices autograd overhead)
      - Tighter SAR clamp ±4 (was ±6) matching ecological-loss bounds
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config  = config
        model_cfg    = config["model"]["eco_transformer"]
        use_validity = config["patches"]["use_validity_mask"]

        self.in_channels  = config["bands"]["count"] + (1 if use_validity else 0)
        self.out_channels = config["bands"]["count"]
        self.t_input  = config["model"]["t_input"]
        self.t_output = config["model"]["t_output"]
        self.sar_idx  = config["bands"]["indices"]["sar_vv"]

        d_model   = model_cfg.get("embed_dim", 128)
        depth     = model_cfg.get("depth", 4)
        n_heads   = model_cfg.get("n_heads", 8)
        mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
        dropout   = model_cfg.get("dropout", 0.1)

        self.spatial_scale = 8
        patch_h = 128 // self.spatial_scale
        patch_w = 128 // self.spatial_scale
        self.n_patches = patch_h * patch_w
        self.patch_h   = patch_h
        self.patch_w   = patch_w
        self.d_model   = d_model

        self.spatial_encoder = SpatialEncoder(
            self.in_channels, d_model, sar_idx=self.sar_idx
        )
        self.spatial_decoder = SpatialDecoder(d_model, self.out_channels)

        self.spatial_pos  = SinusoidalPos2D(d_model, patch_h, patch_w)
        self.temporal_emb = LearnedTemporalEmbedding(d_model, max_len=64)

        self.encoder_blocks = nn.ModuleList([
            FactorizedSTBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.t_output, self.n_patches, d_model)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.cross_attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": PreNormAttention(d_model, n_heads, dropout),
                "self_attn":  PreNormAttention(d_model, n_heads, dropout),
                "ffn":        PreNormFFN(d_model, mlp_ratio, dropout),
            })
            for _ in range(depth // 2)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        feats  = self.spatial_encoder(x_flat)
        D, h, w = feats.shape[1], feats.shape[2], feats.shape[3]
        tokens = feats.reshape(B * T, D, h * w).permute(0, 2, 1)
        tokens = self.spatial_pos(tokens)
        tokens = tokens.reshape(B, T, self.n_patches, D)
        t_indices = torch.arange(T, device=tokens.device)
        t_embs = self.temporal_emb.embedding(t_indices).unsqueeze(0).unsqueeze(2)
        tokens = tokens + t_embs
        return tokens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_frames(x)
        for block in self.encoder_blocks:
            tokens = block(tokens)
        return self.encoder_norm(tokens)

    def decode(self, enc_out: torch.Tensor) -> torch.Tensor:
        B = enc_out.shape[0]
        t_indices = torch.arange(
            self.t_input, self.t_input + self.t_output, device=enc_out.device
        )
        t_embs  = self.temporal_emb.embedding(t_indices).unsqueeze(0).unsqueeze(2)
        queries = self.query_tokens + t_embs
        queries = queries.expand(B, -1, -1, -1)

        B, T_in, N, D = enc_out.shape
        kv = enc_out.reshape(B, T_in * N, D)
        B, T_out, N, D = queries.shape
        q = queries.reshape(B * T_out, N, D)
        kv_expanded = kv.unsqueeze(1).expand(-1, T_out, -1, -1).reshape(B * T_out, T_in * N, D)

        for block in self.cross_attn_blocks:
            q = block["cross_attn"](q, kv_expanded, kv_expanded)
            q = block["self_attn"](q, q, q)
            q = block["ffn"](q)

        return self.decoder_norm(q).reshape(B, T_out, N, D)

    def _decode_to_frames(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, T_out, N, D = tokens.shape
        feats = tokens.reshape(B * T_out, N, D)
        feats = feats.permute(0, 2, 1).reshape(B * T_out, D, self.patch_h, self.patch_w)
        out = self.spatial_decoder(feats)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out.reshape(B, T_out, self.out_channels, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        enc_out = self.encode(x)
        dec_out = self.decode(enc_out)
        preds   = self._decode_to_frames(dec_out, H, W)

        # FIX T1-v3: clamp via torch.stack instead of in-place slice assignment.
        # preds shape: (B, T_out, C, H, W). Clamp per-band with different bounds,
        # then stack. This avoids CopySlices in the autograd graph.
        sar_idx = self.sar_idx
        n       = self.out_channels
        band_list = []
        for c in range(n):
            band = preds[:, :, c]
            if c == sar_idx:
                # FIX T2: tighter SAR clamp (was ±6)
                band = band.clamp(-4.0, 4.0)
            else:
                band = band.clamp(-0.1, 1.1)
            band_list.append(band)
        preds = torch.stack(band_list, dim=2)   # (B, T_out, C, H, W)

        return preds

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_eco_transformer(config: Dict[str, Any]) -> EcoTransformer:
    model = EcoTransformer(config)
    n = model.count_parameters()
    print(f"[EcoTransformer] Parameters: {n:,} ({n/1e6:.1f}M)")
    return model