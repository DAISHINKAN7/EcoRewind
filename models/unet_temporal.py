"""
unet_temporal.py
----------------
U-Net + Temporal Attention model for ECO-REWIND (primary architecture).
 
Design:
  1. Shared spatial encoder (U-Net style): processes all T_in frames by folding
     T into the batch dimension: (B, T, C, H, W) → (B*T, C, H, W)
  2. LSTM at the encoder bottleneck: learns temporal dynamics from spatially
     average-pooled features.
  3. Temporal cross-attention: decoder queries attend to the T_in LSTM outputs
     to selectively weight historical timesteps.
  4. FiLM conditioning: temporal context vector *modulates* the full spatial
     bottleneck features (scale + shift per channel) — replaces the flat
     broadcast that produced zero spatial variation at the bottleneck.
  5. Spatial self-attention at bottleneck: captures global spatial context
     at the coarsest scale (16×16 tokens for 128-px input, very cheap).
  6. Residual output head: model predicts Δ(frame) added to the last input
     frame; easier to learn and gives smoother gradient flow.
 
Improvements over v1:
  - FiLM conditioning on bottleneck (fixes broadcast spatial invariance bug)
  - Bottleneck spatial self-attention
  - Residual decoder (predict delta not absolute)
  - Attention-gated skip aggregation
  - MC Dropout via Dropout2d in every DoubleConv
 
~52M parameters at default config (enc=[32, 64, 128, 256], lstm=512).
Fits on RTX A4000 16GB at batch=4, 128×128 patches.
"""
 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple, Dict, Any, Optional
 
 
# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------
 
class DoubleConv(nn.Module):
    """
    Two conv-BN-GELU blocks with spatial (channel-wise) dropout.
    GELU replaces ReLU — smoother gradient, empirically better for regression.
    """
 
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
 
class Down(nn.Module):
    """MaxPool + DoubleConv."""
 
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
 
class Up(nn.Module):
    """Bilinear upsample + skip connection + DoubleConv."""
 
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, dropout=dropout)
 
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                           0, skip.shape[-2] - x.shape[-2]])
        return self.conv(torch.cat([skip, x], dim=1))
 
 
# ---------------------------------------------------------------------------
# FiLM conditioning
# ---------------------------------------------------------------------------
 
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
 
    Uses a conditioning vector (from LSTM + attention) to scale and shift
    a 2D spatial feature map channel-wise:
 
        out = x * (1 + γ) + β
 
    where γ, β are derived from the conditioning vector.
    This lets temporal context modulate WHERE (spatially) the decoder focuses,
    rather than replacing spatial features with a single broadcast vector.
    """
 
    def __init__(self, d_cond: int, d_spatial: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_cond, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, 2 * d_spatial),
        )
        # Initialise near identity: γ≈0, β≈0 → no modulation at start of training
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
 
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, C, H, W) spatial feature map
        cond : (B, d_cond)  conditioning vector
        """
        params = self.proj(cond)                # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)   # each (B, C)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta  = beta.view(-1, x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta
 
 
# ---------------------------------------------------------------------------
# Spatial self-attention at bottleneck
# ---------------------------------------------------------------------------
 
class BottleneckSpatialAttention(nn.Module):
    """
    Multi-head self-attention over the 2D spatial tokens at the bottleneck.
 
    For a 128×128 input with 3 downs: bottleneck is 16×16 = 256 tokens.
    This is cheap (256×256 attention) but gives the model global spatial context
    — something no stack of 3×3 convolutions can achieve.
    """
 
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        Returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)   # (B, H*W, C)
 
        # Self-attention
        normed = self.norm(tokens)
        attended, _ = self.attn(normed, normed, normed)
        tokens = tokens + attended
 
        # FFN
        tokens = tokens + self.ffn(tokens)
 
        return tokens.permute(0, 2, 1).view(B, C, H, W)
 
 
# ---------------------------------------------------------------------------
# Temporal attention module
# ---------------------------------------------------------------------------
 
class TemporalCrossAttention(nn.Module):
    """
    Multi-head cross-attention over the time axis.
    Query: decoder bottleneck vector at one prediction step.
    Key/Value: all T_in LSTM encoder outputs.
    """
 
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)
 
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)
 
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query   : (B, d_model)
        context : (B, T, d_model)
        Returns : (B, d_model)
        """
        residual = query
        B, T, D = context.shape
 
        q = self.q_proj(query).unsqueeze(1)   # (B, 1, D)
        k = self.k_proj(context)              # (B, T, D)
        v = self.v_proj(context)
 
        q = rearrange(q, "b 1 (h d) -> b h 1 d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)
 
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
 
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h 1 d -> b (h d)")
        out = self.out_proj(out)
        return self.norm(out + residual)
 
 
# ---------------------------------------------------------------------------
# Spatial Encoder
# ---------------------------------------------------------------------------
 
class UNetEncoder(nn.Module):
    def __init__(self, in_ch: int, encoder_channels: List[int], dropout: float = 0.1):
        super().__init__()
        self.inc  = DoubleConv(in_ch, encoder_channels[0], dropout=dropout)
        self.downs = nn.ModuleList([
            Down(encoder_channels[i-1], encoder_channels[i], dropout=dropout)
            for i in range(1, len(encoder_channels))
        ])
 
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x : (B*T, C, H, W) → list of skip tensors, finest → coarsest"""
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        return skips
 
 
# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
 
class UNetTemporalModel(nn.Module):
    """
    Improved U-Net + Temporal Attention model for counterfactual prediction.
 
    MC Dropout: call enable_mc_dropout() before inference then run N passes.
    """
 
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_cfg   = config["model"]["unet_temporal"]
        use_validity = config["patches"]["use_validity_mask"]
 
        self.in_channels  = config["bands"]["count"] + (1 if use_validity else 0)
        self.out_channels = config["bands"]["count"]
        self.t_input  = config["model"]["t_input"]
        self.t_output = config["model"]["t_output"]
 
        enc_channels  = model_cfg["encoder_channels"]   # [32, 64, 128, 256]
        lstm_hidden   = model_cfg["lstm_hidden"]         # 512
        n_heads       = model_cfg["n_attention_heads"]   # 8
        dropout       = model_cfg.get("dropout", 0.1)
 
        self.bottleneck_ch = enc_channels[-1]
 
        # ---- Encoder ----
        self.encoder = UNetEncoder(self.in_channels, enc_channels, dropout=dropout)
 
        # ---- Bottleneck spatial self-attention ----
        # Operates on (B*T, C_bn, H_bn, W_bn) after folding T into batch.
        # H_bn = W_bn = 16 for 128px input → 256 tokens → cheap.
        self.bottleneck_attn = BottleneckSpatialAttention(
            d_model=self.bottleneck_ch,
            n_heads=min(4, self.bottleneck_ch // 16),
            dropout=dropout,
        )
 
        # ---- Temporal LSTM ----
        self.lstm = nn.LSTM(
            input_size=self.bottleneck_ch,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm_proj = nn.Linear(lstm_hidden, self.bottleneck_ch)
 
        # ---- Temporal cross-attention for decoder queries ----
        self.temporal_attn = TemporalCrossAttention(self.bottleneck_ch, n_heads, dropout)
 
        # ---- FiLM: conditions spatial bottleneck on temporal context ----
        self.film = FiLMLayer(
            d_cond=self.bottleneck_ch,
            d_spatial=self.bottleneck_ch,
        )
 
        # ---- Decoder ----
        dec_channels = list(reversed(enc_channels[:-1]))   # [128, 64, 32]
        self.decoder_ups = nn.ModuleList()
        in_ch = self.bottleneck_ch
        for i, out_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i+2)]
            self.decoder_ups.append(Up(in_ch, skip_ch, out_ch, dropout=dropout))
            in_ch = out_ch
 
        # ---- Output head — predicts residual Δ from last input frame ----
        # Tanh output: delta in (−1, 1); added to last_frame via scaled residual
        self.output_head = nn.Sequential(
            nn.Conv2d(dec_channels[-1], dec_channels[-1] // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dec_channels[-1] // 2), dec_channels[-1] // 2),
            nn.GELU(),
            nn.Conv2d(dec_channels[-1] // 2, self.out_channels, 1),
            nn.Tanh(),
        )
        # Learned scale parameter for the residual.
        # sigmoid(0) = 0.5 → max 0.5 change per step. Initialised to 0 → small Δ at start.
        self.delta_scale = nn.Parameter(torch.zeros(1))
 
        # Positional embedding for decoder timesteps
        self.pos_embed = nn.Embedding(self.t_output + self.t_input + 10, self.bottleneck_ch)
 
        self.dropout_layer = nn.Dropout(dropout)
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    def enable_mc_dropout(self):
        """Enable MC Dropout at inference: keep Dropout layers active."""
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T_in, C, H, W)
        Returns:
            (B, T_out, C_out, H, W)
        """
        B, T_in, C, H, W = x.shape
 
        # Last input frame — used as residual base
        last_frame = x[:, -1, :self.out_channels]   # (B, C_out, H, W)
 
        # ---- ENCODE: fold T into batch dimension ----
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        skips_flat = self.encoder(x_flat)   # list of (B*T, ch, H_i, W_i)
 
        # ---- BOTTLENECK SPATIAL SELF-ATTENTION ----
        # Apply after the deepest encoder level.
        # Gives global spatial context before temporal aggregation.
        bottleneck_flat = self.bottleneck_attn(skips_flat[-1])   # (B*T, C_bn, H_bn, W_bn)
        skips_flat[-1] = bottleneck_flat
 
        C_bn = bottleneck_flat.shape[1]
        H_bn, W_bn = bottleneck_flat.shape[2], bottleneck_flat.shape[3]
 
        # ---- TEMPORAL LSTM on pooled bottleneck ----
        pooled = bottleneck_flat.mean(dim=(-2, -1))                      # (B*T, C_bn)
        pooled = rearrange(pooled, "(b t) c -> b t c", b=B, t=T_in)     # (B, T, C_bn)
        lstm_out, _ = self.lstm(pooled)                                  # (B, T, lstm_hidden)
        temporal_context = self.lstm_proj(lstm_out)                      # (B, T, C_bn)
 
        # Spatial bottleneck (averaged over T_in): preserves spatial structure
        spatial_bottleneck = rearrange(
            bottleneck_flat, "(b t) c h w -> b t c h w", b=B, t=T_in
        ).mean(dim=1)   # (B, C_bn, H_bn, W_bn)
 
        # Reshape skips for aggregation
        skips_bt = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B, t=T_in)
            for s in skips_flat
        ]
        # Temporal-mean skip aggregation (captures full pre-event baseline)
        skips_mean = [s.mean(dim=1) for s in skips_bt]   # (B, ch, H_i, W_i) each
 
        # Initial decoder state = last LSTM output
        dec_state = temporal_context[:, -1]   # (B, C_bn)
 
        # ---- DECODE: generate T_out steps ----
        preds = []
 
        for step in range(self.t_output):
            # 1. Temporal cross-attention: which historical frames matter most
            pos     = torch.tensor([T_in + step], device=x.device)
            pos_emb = self.pos_embed(pos).squeeze(0)          # (C_bn,)
            query   = dec_state + pos_emb.unsqueeze(0)        # (B, C_bn)
            attended = self.temporal_attn(query, temporal_context)   # (B, C_bn)
 
            # 2. FiLM: condition spatial bottleneck on temporal context
            #    This replaces the old "broadcast attended to (B, C, H, W)" which
            #    had zero spatial variation — every pixel got the same vector.
            spatial_bn = self.film(spatial_bottleneck, attended)   # (B, C_bn, H_bn, W_bn)
 
            # 3. Spatial decoder with U-Net skip connections
            feat = spatial_bn
            for i, up in enumerate(self.decoder_ups):
                skip = skips_mean[-(i+2)]   # finest skips last
                feat = up(feat, skip)       # (B, out_ch, H_i, W_i)
 
            # 4. Residual output: predict Δ, add to last input frame
            delta = self.output_head(feat)                          # (B, C, H, W), Tanh
            scale = torch.sigmoid(self.delta_scale) * 0.5          # [0, 0.5]
            pred  = last_frame + delta * scale                      # (B, C, H, W)
            # Don't clamp here — ecological loss handles out-of-range penalisation
 
            preds.append(pred)
 
            # Update decoder state for next step
            dec_state = attended
 
        return torch.stack(preds, dim=1)   # (B, T_out, C_out, H, W)
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)