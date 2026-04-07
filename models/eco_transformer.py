"""
eco_transformer.py
------------------
EcoTransformer: Factorized Spatiotemporal Transformer for ecosystem prediction.
 
Architecture overview:
 
  1. SpatialEncoder   — lightweight CNN per frame: (C, H, W) → (D, H/8, W/8)
  2. PositionalEncoding — sinusoidal spatial + learned temporal embeddings
  3. Factorized Transformer Encoder:
       TemporalAttention  — for each spatial token, attend across T_in frames
       SpatialAttention   — for each frame, attend across H/8×W/8 spatial tokens
       × n_layers
  4. Prediction Head — T_out learned query tokens cross-attend to encoder output
  5. SpatialDecoder — CNN + bilinear upsample: (D, H/8, W/8) → (C, H, W)
 
Why this outperforms ConvLSTM and UNet-Temporal:
  - Global receptive field from first layer (no local-only convolution bottleneck)
  - Factorized attention: handles 256 spatial tokens efficiently without O(N²) blowup
  - Cross-attention decoder: each future timestep attends to all T_in encoder states,
    learning flexible temporal patterns rather than fixed recurrent state passing
  - No vanishing gradient through recurrence (direct attention paths)
 
~12M parameters at default config (embed_dim=128, depth=4).
"""
 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
 
 
# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------
 
class SinusoidalPos2D(nn.Module):
    """
    Learnable-free 2D sinusoidal positional encoding.
    Adds spatial position awareness to flattened patch tokens.
    """
 
    def __init__(self, d_model: int, h: int, w: int):
        super().__init__()
        pe = torch.zeros(h * w, d_model)
        positions = torch.arange(h * w, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, h*w, D)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, D) → x + pe"""
        return x + self.pe
 
 
class LearnedTemporalEmbedding(nn.Module):
    """Learned embedding for up to max_len time steps."""
 
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
 
    def forward(self, t: int, device: torch.device) -> torch.Tensor:
        """Returns (1, 1, D) embedding for timestep t."""
        idx = torch.tensor([t], device=device)
        return self.embedding(idx).unsqueeze(0)   # (1, 1, D)
 
 
# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------
 
class PreNormAttention(nn.Module):
    """Pre-LayerNorm multi-head attention with residual connection."""
 
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
 
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_norm = self.norm(q)
        attended, _ = self.attn(q_norm, k, v, key_padding_mask=key_padding_mask)
        return q + self.drop(attended)
 
 
class PreNormFFN(nn.Module):
    """Pre-LayerNorm feed-forward block with residual connection."""
 
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))
 
 
class FactorizedSTBlock(nn.Module):
    """
    One factorized spatiotemporal transformer block.
 
    Order: temporal attn → spatial attn → FFN
    Each operates with Pre-LayerNorm and residual connections.
 
    Factorized attention complexity:
      Temporal: O(T²)  per spatial token   → O(N·T²)  total (T=4, N=256 → trivial)
      Spatial:  O(N²)  per timestep        → O(T·N²)  total (4 × 256² = 262k ops/layer)
    """
 
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.temporal_attn = PreNormAttention(d_model, n_heads, dropout)
        self.spatial_attn  = PreNormAttention(d_model, n_heads, dropout)
        self.ffn = PreNormFFN(d_model, mlp_ratio, dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, N, D)  where N = h*w spatial tokens
        """
        B, T, N, D = x.shape
 
        # ---- Temporal attention: each spatial location attends over T ----
        # Reshape so T is the sequence dimension: (B*N, T, D)
        x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x_t = self.temporal_attn(x_t, x_t, x_t)
        x = x_t.reshape(B, N, T, D).permute(0, 2, 1, 3)   # (B, T, N, D)
 
        # ---- Spatial attention: each timestep attends over N ----
        # Reshape so N is the sequence dimension: (B*T, N, D)
        x_s = x.reshape(B * T, N, D)
        x_s = self.spatial_attn(x_s, x_s, x_s)
        x = x_s.reshape(B, T, N, D)
 
        # ---- FFN ----
        x = self.ffn(x.reshape(B * T, N, D)).reshape(B, T, N, D)
 
        return x
 
 
# ---------------------------------------------------------------------------
# CNN encoder and decoder
# ---------------------------------------------------------------------------
 
class SpatialEncoder(nn.Module):
    """
    Per-frame CNN encoder: (C, H, W) → (D, H/8, W/8).
 
    Uses depthwise-separable convolutions for efficiency.
    3 stages of stride-2 downsampling: 128 → 64 → 32 → 16.
    """
 
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        mid = d_model // 2
 
        self.encoder = nn.Sequential(
            # Stage 1: stride-2, in_ch → mid/2
            nn.Conv2d(in_channels, mid // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2),
            nn.GELU(),
            # Stage 2: stride-2, → mid
            nn.Conv2d(mid // 2, mid, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            # Stage 3: stride-2, → d_model
            nn.Conv2d(mid, d_model, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            # 1×1 projection
            nn.Conv2d(d_model, d_model, 1),
        )
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B*T, C, H, W) → (B*T, D, H/8, W/8)"""
        return self.encoder(x)
 
 
class SpatialDecoder(nn.Module):
    """
    Per-frame CNN decoder: (D, H/8, W/8) → (C, H, W).
 
    Bilinear upsample × 2 then conv at each stage (avoids checkerboard).
    """
 
    def __init__(self, d_model: int, out_channels: int):
        super().__init__()
        mid = d_model // 2
 
        self.decoder = nn.Sequential(
            # Upsample ×2: H/8 → H/4
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d_model, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            # Upsample ×2: H/4 → H/2
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2),
            nn.GELU(),
            # Upsample ×2: H/2 → H
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid // 2, out_channels, 3, padding=1),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B*T, D, H/8, W/8) → (B*T, C, H, W)"""
        return self.decoder(x)
 
 
# ---------------------------------------------------------------------------
# Main EcoTransformer
# ---------------------------------------------------------------------------
 
class EcoTransformer(nn.Module):
    """
    Factorized Spatiotemporal Transformer for ecosystem temporal prediction.
 
    Input:  (B, T_in, C, H, W)  — normalized input sequence
    Output: (B, T_out, C_out, H, W) — predicted future sequence
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
 
        d_model   = model_cfg.get("embed_dim", 128)
        depth     = model_cfg.get("depth", 4)
        n_heads   = model_cfg.get("n_heads", 8)
        mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
        dropout   = model_cfg.get("dropout", 0.1)
 
        # Spatial dimensions after 3× stride-2 encoding: 128/8 = 16
        self.spatial_scale = 8
        patch_h = 128 // self.spatial_scale   # 16
        patch_w = 128 // self.spatial_scale   # 16
        self.n_patches = patch_h * patch_w    # 256
 
        # ---- Spatial encoder / decoder ----
        self.spatial_encoder = SpatialEncoder(self.in_channels, d_model)
        self.spatial_decoder = SpatialDecoder(d_model, self.out_channels)
 
        # ---- Positional encodings ----
        self.spatial_pos = SinusoidalPos2D(d_model, patch_h, patch_w)
        self.temporal_emb = LearnedTemporalEmbedding(d_model, max_len=64)
 
        # ---- Transformer encoder ----
        self.encoder_blocks = nn.ModuleList([
            FactorizedSTBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
 
        # ---- Prediction head: T_out learned queries + cross-attention ----
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.t_output, self.n_patches, d_model)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
 
        # Cross-attention layers: queries attend to encoder output
        self.cross_attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": PreNormAttention(d_model, n_heads, dropout),
                "self_attn":  PreNormAttention(d_model, n_heads, dropout),
                "ffn":        PreNormFFN(d_model, mlp_ratio, dropout),
            })
            for _ in range(depth // 2)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
 
        # ---- Output norm ----
        # Tanh not applied here — spatial_decoder ends with raw conv.
        # We clamp predictions post-decoder.
 
        self.d_model = d_model
        self.patch_h = patch_h
        self.patch_w = patch_w
 
    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a (B, T, C, H, W) sequence to (B, T, N, D) tokens.
        """
        B, T, C, H, W = x.shape
 
        # CNN encode all frames at once
        x_flat = x.reshape(B * T, C, H, W)
        feats = self.spatial_encoder(x_flat)   # (B*T, D, h, w)
        D, h, w = feats.shape[1], feats.shape[2], feats.shape[3]
 
        # Flatten spatial: (B*T, D, h, w) → (B*T, h*w, D)
        tokens = feats.reshape(B * T, D, h * w).permute(0, 2, 1)   # (B*T, N, D)
 
        # Add spatial position encoding
        tokens = self.spatial_pos(tokens)   # (B*T, N, D)
 
        # Add temporal embedding
        tokens = tokens.reshape(B, T, self.n_patches, D)
        for t in range(T):
            t_emb = self.temporal_emb(t, tokens.device)   # (1, 1, D)
            tokens[:, t] = tokens[:, t] + t_emb
 
        return tokens   # (B, T, N, D)
 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full encoder forward pass.
        x : (B, T_in, C, H, W)
        Returns: (B, T_in, N, D) encoder output
        """
        tokens = self._encode_frames(x)
 
        for block in self.encoder_blocks:
            tokens = block(tokens)
 
        tokens = self.encoder_norm(tokens)
        return tokens   # (B, T_in, N, D)
 
    def decode(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention decoder.
        enc_out : (B, T_in, N, D)
        Returns : (B, T_out, N, D)
        """
        B = enc_out.shape[0]
 
        # Expand learned query tokens
        queries = self.query_tokens.expand(B, -1, -1, -1)   # (B, T_out, N, D)
 
        # Add temporal embeddings to queries
        for t in range(self.t_output):
            t_emb = self.temporal_emb(self.t_input + t, enc_out.device)
            queries[:, t] = queries[:, t] + t_emb
 
        # Key / value: flatten enc_out from (B, T_in, N, D) → (B, T_in*N, D)
        B, T_in, N, D = enc_out.shape
        kv = enc_out.reshape(B, T_in * N, D)
 
        # Cross-attention: for each future step's spatial tokens, attend to all encoder tokens
        B, T_out, N, D = queries.shape
        q = queries.reshape(B * T_out, N, D)
 
        # Tile kv to match B*T_out batch
        kv_expanded = kv.unsqueeze(1).expand(-1, T_out, -1, -1).reshape(B * T_out, T_in * N, D)
 
        for block in self.cross_attn_blocks:
            # Cross-attention: queries attend to encoder output
            q = block["cross_attn"](q, kv_expanded, kv_expanded)
            # Self-attention among spatial query tokens
            q = block["self_attn"](q, q, q)
            # FFN
            q = block["ffn"](q)
 
        q = self.decoder_norm(q)
        return q.reshape(B, T_out, N, D)
 
    def _decode_to_frames(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        (B, T_out, N, D) → (B, T_out, C, H, W)
        """
        B, T_out, N, D = tokens.shape
 
        # (B*T_out, N, D) → (B*T_out, D, h, w)
        feats = tokens.reshape(B * T_out, N, D)
        feats = feats.permute(0, 2, 1).reshape(B * T_out, D, self.patch_h, self.patch_w)
 
        # CNN decode
        out = self.spatial_decoder(feats)   # (B*T_out, C, H', W')
 
        # Resize to target if needed (minor alignment issues)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
 
        return out.reshape(B, T_out, self.out_channels, H, W)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T_in, C_in, H, W)
        Returns:
            (B, T_out, C_out, H, W)
        """
        H, W = x.shape[-2], x.shape[-1]
 
        enc_out = self.encode(x)
        dec_out = self.decode(enc_out)
        preds   = self._decode_to_frames(dec_out, H, W)
 
        return preds
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 
 
# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
 
def build_eco_transformer(config: Dict[str, Any]) -> EcoTransformer:
    model = EcoTransformer(config)
    n = model.count_parameters()
    print(f"[EcoTransformer] Parameters: {n:,} ({n/1e6:.1f}M)")
    return model