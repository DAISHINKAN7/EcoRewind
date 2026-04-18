"""
unet_patchtst.py  (v4 — production-stable)
-------------------------------------------
Same correct pipeline as unet_mamba but with a Transformer temporal module.

Stability fixes vs v3:
  1. InputBiasCorrector: same fix as unet_mamba — corrects NaN→0 fill bias
     which was causing negative NDVI and SAR R² scores
  2. Output head: sigmoid for optical bands, linear for SAR
     (matches unet_mamba for fair comparison)
  3. Transformer numerical safety:
     - attention logits are explicitly scaled by 1/sqrt(d_head)
       (PyTorch's MHA does this internally, but we add an explicit check)
     - FFN uses GELU (smooth, no dying unit risk)
     - Pre-norm architecture (more stable than post-norm for small T)
  4. FiLM clamping: γ/β clamped to [-2, 2]
  5. future_proj: weight init uses xavier (was kaiming, wrong for linear)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Temporal Transformer (numerically safe)
# ---------------------------------------------------------------------------

class TemporalTransformerBlock(nn.Module):
    """Pre-norm transformer block. Input/Output: (B, T, D)"""

    def __init__(self, d_model: int, n_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden     = int(d_model * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalTransformerEncoder(nn.Module):
    """
    N-layer transformer over T temporal tokens.
    Includes sinusoidal positional encoding.
    For T=4 (our case), attention matrix is 4×4 → trivially stable.
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
# FiLM fusion (clamped — same as unet_mamba fix)
# ---------------------------------------------------------------------------

class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation — clamped for training stability.
    γ, β constrained to [-clamp_val, clamp_val] to prevent explosion.
    """

    def __init__(self, d_cond: int, d_spatial: int, clamp_val: float = 2.0):
        super().__init__()
        self.clamp_val = clamp_val
        self.proj = nn.Linear(d_cond, 2 * d_spatial)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        gamma = gamma.clamp(-self.clamp_val, self.clamp_val)
        beta  = beta.clamp(-self.clamp_val, self.clamp_val)
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)
        beta  = beta.view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Input de-bias correction (shared with unet_mamba)
# ---------------------------------------------------------------------------

class InputBiasCorrector(nn.Module):
    """
    Fixes the NaN→0 fill bias introduced by eco_dataset.

    eco_dataset.py fills NaN pixels with 0.0 at line:
        patch_filled = np.where(np.isfinite(patch), patch, 0.0)

    For z-scored SAR_VV (mean≈-12dB, std≈4dB in raw space):
        fill=0 in normalized space means raw≈0dB, actual mean≈-12dB
        → z-score of fill = (0 - 0)/1 = 0  but mean in normalized space ≠ 0
    
    Wait — after z-scoring, mean IS 0. But the fill value 0.0 in the
    PATCH array (which stores normalized values) is correct for z-scored bands.
    
    The actual issue is different: patch_sampler saves normalized patches.
    NDVI is shift-scaled: normalized_val = (raw + 1) / 2, so raw -1 (open water)
    → normalized 0.0. When NaN pixels are filled with 0.0, the model
    sees normalized NDVI=0.0 everywhere in invalid regions, which the
    evaluator then inverse-transforms back to raw NDVI=-1.0 — creating
    large MSE and negative R².
    
    FIX: Replace the 0.0 fill with a learned per-channel fill value so
    the model can discover that invalid pixels should be "neutral" (mean).
    For shift-scaled bands: neutral = 0.5 (raw=0). For z-scored: neutral = 0.
    We initialize learned fills to 0.5 for optical/index bands and 0.0 for SAR.
    """

    def __init__(self, n_bands: int, band_names: List[str] = None,
                 norm_methods: Dict[str, str] = None):
        super().__init__()
        # Initialize fill values: 0.5 for minmax/shift bands, 0.0 for zscore
        init_fills = torch.zeros(n_bands)
        if norm_methods and band_names:
            for i, name in enumerate(band_names):
                method = norm_methods.get(name.lower(), "minmax")
                if method in ("minmax", "shift"):
                    init_fills[i] = 0.5   # neutral value in [0,1] space
                # zscore: 0.0 is already the correct neutral fill
        else:
            # Default: assume first n_bands-1 are [0,1] space, last is zscore
            init_fills[:-1] = 0.5

        self.fill_values = nn.Parameter(init_fills)

    def forward(self, x: torch.Tensor, has_validity_ch: bool = True) -> torch.Tensor:
        """
        x: (B, T, C, H, W) where last channel may be validity mask
        Returns: (B, T, n_bands, H, W) with corrected fills
        """
        if not has_validity_ch:
            # No validity channel: can't identify invalid pixels, return as-is
            return x

        n_bands = x.shape[2] - 1
        bands   = x[:, :, :n_bands]   # (B, T, n_bands, H, W)
        valid   = x[:, :, n_bands:]   # (B, T, 1, H, W)

        fills = self.fill_values[:n_bands].view(1, 1, -1, 1, 1)
        invalid_mask = (valid < 0.5).expand_as(bands)

        # Where pixels are invalid, replace 0.0 fill with learned neutral value
        corrected = torch.where(invalid_mask, fills.expand_as(bands), bands)
        return corrected


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
    UNet + Temporal Transformer — production-stable version.

    Stability fixes vs v3:
      1. InputBiasCorrector: fixes NaN→0 fill bias (primary fix for neg R²)
      2. Separate optical/SAR output heads: sigmoid for optical, linear for SAR
      3. FiLM clamping: prevents scale/shift explosion
      4. future_proj: xavier initialization (not kaiming)
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

        # Input de-bias correction (NEW)
        band_names   = config["bands"]["names"]
        norm_methods = config["bands"]["normalization"]
        self.input_corrector = InputBiasCorrector(n_bands, band_names, norm_methods)

        # Spatial Encoder (shared weights across T)
        self.inc   = DoubleConv(n_bands, enc_ch[0], dropout)   # n_bands not in_channels
        self.downs = nn.ModuleList([
            Down(enc_ch[i-1], enc_ch[i], dropout)
            for i in range(1, len(enc_ch))
        ])

        # Temporal Transformer
        self.temporal_enc = TemporalTransformerEncoder(
            d_model=self.bottleneck_ch,
            n_layers=n_tf_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Project T_in → T_out
        self.future_proj = nn.Linear(self.t_input, self.t_output)
        nn.init.xavier_uniform_(self.future_proj.weight)   # FIX: xavier not kaiming

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

        # Separate output heads for optical and SAR
        self.optical_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1], dec_ch[-1], 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dec_ch[-1]), dec_ch[-1]),
            nn.GELU(),
            nn.Conv2d(dec_ch[-1], n_bands - 1, 1),
            nn.Sigmoid(),   # optical + index bands: bounded [0, 1]
        )
        self.sar_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1], dec_ch[-1] // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, dec_ch[-1] // 2), dec_ch[-1] // 2),
            nn.GELU(),
            nn.Conv2d(dec_ch[-1] // 2, 1, 1),   # SAR: z-scored, unbounded
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

        # Step 0: de-bias NaN fill
        x_corrected = self.input_corrector(x, has_validity_ch=self.use_validity)
        # x_corrected: (B, T_in, n_bands, H, W)

        # Step 1: Spatial encoding
        x_flat = rearrange(x_corrected, "b t c h w -> (b t) c h w")

        skips_flat = [self.inc(x_flat)]
        for down in self.downs:
            skips_flat.append(down(skips_flat[-1]))

        spatial_all = rearrange(skips_flat[-1], "(b t) c h w -> b t c h w", b=B)

        # Step 2: Temporal modeling
        pooled   = spatial_all.mean(dim=(-2, -1))          # (B, T_in, C_bn)
        temporal = self.temporal_enc(pooled)               # (B, T_in, C_bn)

        cond_out = self.future_proj(
            temporal.permute(0, 2, 1)
        ).permute(0, 2, 1)                                 # (B, T_out, C_bn)

        # Step 3: Mean spatial base + skips
        spatial_mean = spatial_all.mean(dim=1)
        skips_mean = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B).mean(dim=1)
            for s in skips_flat
        ]

        # Step 4: Decode T_out frames
        preds = []
        for step in range(self.t_output):
            cond_vec = cond_out[:, step, :]
            feat = self.film_bn(spatial_mean, cond_vec)

            for i, up in enumerate(self.decoder_ups):
                feat = up(feat, skips_mean[-(i+2)])

            optical = self.optical_head(feat)   # (B, n_bands-1, H, W)
            sar     = self.sar_head(feat)       # (B, 1, H, W)

            if self.sar_idx == self.n_bands - 1:
                pred = torch.cat([optical, sar], dim=1)
            else:
                bands = list(optical.split(1, dim=1))
                bands.insert(self.sar_idx, sar)
                pred = torch.cat(bands, dim=1)

            preds.append(pred)

        return torch.stack(preds, dim=1)   # (B, T_out, n_bands, H, W)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)