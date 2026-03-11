"""
unet_temporal.py
----------------
U-Net + Temporal Attention model for ECO-REWIND (primary architecture).
 
Design:
  1. Shared spatial encoder (U-Net style): processes all T_in frames together
     by reshaping (B, T, C, H, W) → (B*T, C, H, W)
  2. LSTM at the encoder bottleneck: learns temporal dynamics
  3. Temporal cross-attention: decoder queries temporal context from encoder
  4. Spatial decoder: upsamples back to (B, T_out, C, H, W) with skip connections
 
~50M parameters at default config.
Fits on RTX A4000 16GB at batch=4, 128×128 patches (~8–10 GB VRAM).
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
    """Two conv-BN-ReLU blocks (standard U-Net unit)."""
 
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
 
class Down(nn.Module):
    """MaxPool + DoubleConv."""
 
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
 
class Up(nn.Module):
    """Bilinear upsample + skip connection + DoubleConv."""
 
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)
 
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims differ (odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])
        return self.conv(torch.cat([skip, x], dim=1))
 
 
# ---------------------------------------------------------------------------
# Temporal attention module
# ---------------------------------------------------------------------------
 
class TemporalCrossAttention(nn.Module):
    """
    Multi-head cross-attention over the time axis.
 
    Query: decoder feature at current step
    Key/Value: all T_in encoder bottleneck features
 
    Allows decoder to selectively attend to the most relevant
    historical timesteps when generating each prediction step.
    """
 
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
 
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
 
    def forward(
        self, query: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query   : (B, d_model) — decoder query at one timestep
            context : (B, T, d_model) — encoder temporal sequence
 
        Returns:
            (B, d_model) — attended output
        """
        residual = query
        B, T, D = context.shape
 
        q = self.q_proj(query).unsqueeze(1)   # (B, 1, D)
        k = self.k_proj(context)              # (B, T, D)
        v = self.v_proj(context)              # (B, T, D)
 
        # Reshape to multi-head
        q = rearrange(q, "b 1 (h d) -> b h 1 d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)
 
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, h, 1, T)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
 
        out = torch.matmul(attn, v)   # (B, h, 1, d)
        out = rearrange(out, "b h 1 d -> b (h d)")
 
        out = self.out_proj(out)
        return self.norm(out + residual)
 
 
# ---------------------------------------------------------------------------
# Spatial Encoder
# ---------------------------------------------------------------------------
 
class UNetEncoder(nn.Module):
    """
    U-Net spatial encoder. Processes all timesteps simultaneously
    by folding T into the batch dimension.
    """
 
    def __init__(self, in_ch: int, encoder_channels: List[int]):
        super().__init__()
        # encoder_channels: e.g. [32, 64, 128, 256]
        self.inc = DoubleConv(in_ch, encoder_channels[0])
        self.downs = nn.ModuleList()
        for i in range(1, len(encoder_channels)):
            self.downs.append(Down(encoder_channels[i-1], encoder_channels[i]))
 
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x : (B*T, C, H, W)
        Returns:
            skips : list of feature maps at each scale, finest → coarsest
        """
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        return skips
 
 
# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
 
class UNetTemporalModel(nn.Module):
    """
    U-Net + Temporal Attention model for counterfactual trajectory prediction.
    """
 
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_cfg = config["model"]["unet_temporal"]
        use_validity = config["patches"]["use_validity_mask"]
 
        self.in_channels = config["bands"]["count"] + (1 if use_validity else 0)
        self.out_channels = config["bands"]["count"]
        self.t_input = config["model"]["t_input"]
        self.t_output = config["model"]["t_output"]
 
        enc_channels = model_cfg["encoder_channels"]   # [32, 64, 128, 256]
        lstm_hidden = model_cfg["lstm_hidden"]          # 512
        n_heads = model_cfg["n_attention_heads"]        # 8
        dropout = model_cfg.get("dropout", 0.1)
 
        self.bottleneck_ch = enc_channels[-1]
 
        # Spatial encoder
        self.encoder = UNetEncoder(self.in_channels, enc_channels)
 
        # Temporal LSTM at bottleneck
        # Bottleneck features are spatially average-pooled → (B, T, C_bn)
        self.lstm = nn.LSTM(
            input_size=self.bottleneck_ch,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        # Project LSTM output back to bottleneck_ch
        self.lstm_proj = nn.Linear(lstm_hidden, self.bottleneck_ch)
 
        # Temporal cross-attention for decoder
        self.temporal_attn = TemporalCrossAttention(self.bottleneck_ch, n_heads, dropout)
 
        # Decoder — one Up block per encoder level (reversed)
        # Skip channels: enc_channels in reverse (excluding bottleneck)
        # Decoder channels: mirror of encoder
        dec_channels = list(reversed(enc_channels[:-1]))  # [128, 64, 32]
        self.decoder_ups = nn.ModuleList()
        in_ch = self.bottleneck_ch
        for i, out_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i+2)]  # matching encoder skip
            self.decoder_ups.append(Up(in_ch, skip_ch, out_ch))
            in_ch = out_ch
 
        # Final output head
        self.output_head = nn.Conv2d(dec_channels[-1], self.out_channels, 1)
 
        # Positional encoding for decoder timesteps
        self.pos_embed = nn.Embedding(self.t_output + self.t_input + 10, self.bottleneck_ch)
 
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T_in, C, H, W)
        Returns:
            (B, T_out, C_out, H, W)
        """
        B, T_in, C, H, W = x.shape
 
        # --- ENCODE: fold T into batch dimension ---
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        skips_flat = self.encoder(x_flat)   # list of (B*T, ch, H_i, W_i)
 
        # --- TEMPORAL LSTM on pooled bottleneck features ---
        bottleneck_flat = skips_flat[-1]    # (B*T, C_bn, H_bn, W_bn)
        C_bn = bottleneck_flat.shape[1]
        H_bn, W_bn = bottleneck_flat.shape[2], bottleneck_flat.shape[3]
 
        # Global average pool → (B*T, C_bn) → (B, T, C_bn)
        pooled = bottleneck_flat.mean(dim=(-2, -1))
        pooled = rearrange(pooled, "(b t) c -> b t c", b=B, t=T_in)
 
        # LSTM temporal encoding
        lstm_out, (h_n, c_n) = self.lstm(pooled)    # (B, T, lstm_hidden)
        lstm_out = self.lstm_proj(lstm_out)          # (B, T, C_bn)
 
        # lstm_out is the temporal context for attention
        temporal_context = lstm_out   # (B, T_in, C_bn)
 
        # --- DECODE: generate T_out steps ---
        preds = []
 
        # Reshape skips back to (B, T, ch, H_i, W_i)
        skips_bt = [
            rearrange(s, "(b t) c h w -> b t c h w", b=B, t=T_in)
            for s in skips_flat
        ]
        # Use last-timestep skips for decoder (most recent context)
        skips_last = [s[:, -1] for s in skips_bt]   # list of (B, ch, H_i, W_i)
 
        # Initial decoder bottleneck = last LSTM output
        dec_bottleneck = lstm_out[:, -1]   # (B, C_bn)
 
        for step in range(self.t_output):
            # Attend to temporal context
            pos = torch.tensor([T_in + step], device=x.device)
            pos_emb = self.pos_embed(pos).squeeze(0)          # (C_bn,)
            query = dec_bottleneck + pos_emb.unsqueeze(0)     # (B, C_bn)
            attended = self.temporal_attn(query, temporal_context)  # (B, C_bn)
 
            # Reshape attended back to spatial bottleneck
            # Broadcast to spatial dims of bottleneck
            spatial_bn = attended.unsqueeze(-1).unsqueeze(-1).expand(
                B, C_bn, H_bn, W_bn
            )   # (B, C_bn, H_bn, W_bn)
 
            # Decode with skip connections
            feat = spatial_bn
            skip_idx = len(self.decoder_ups)
            for i, up in enumerate(self.decoder_ups):
                # Skip from encoder: index from bottleneck backwards
                skip = skips_last[-(i+2)]   # coarser → finer
                feat = up(feat, skip)
 
            # Final prediction
            pred = self.output_head(feat)   # (B, C_out, H, W)
            preds.append(pred)
 
            # Update decoder bottleneck (append prediction to context implicitly)
            dec_bottleneck = attended
 
        return torch.stack(preds, dim=1)   # (B, T_out, C_out, H, W)
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)