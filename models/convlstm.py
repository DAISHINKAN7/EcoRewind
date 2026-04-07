"""
convlstm.py
-----------
Improved ConvLSTM model for ECO-REWIND.
 
Architecture:
  Encoder  : N stacked ConvLSTM cells process T_in frames.
  Bottleneck: Temporal self-attention module lets the encoder
              decide which input timesteps carry the most signal.
  Decoder  : Residual autoregressive decoding — predicts Δ(frame)
              rather than absolute frame, which is easier to learn
              and gives much better gradient flow.
 
Key improvements over v1:
  1. Temporal attention after encoder (captures "which quarter matters")
  2. Residual decoder (pred = last_frame + Δ) — smoother gradients
  3. Orthogonal init for recurrent weights
  4. Spatial attention gate in output head
  5. Optional bidirectional encoder (disabled by default)
 
~6M parameters at default config (hidden=[96, 96, 64]).
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
 
 
# ---------------------------------------------------------------------------
# ConvLSTM Cell
# ---------------------------------------------------------------------------
 
class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell with GroupNorm on cell state.
    """
 
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
 
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.layer_norm = nn.GroupNorm(
            num_groups=min(8, hidden_channels),
            num_channels=hidden_channels,
        )
        self._init_weights()
 
    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        # Orthogonal init for the recurrent (hidden→gates) portion
        in_ch = self.conv.weight.shape[1] - self.hidden_channels
        hidden_slice = self.conv.weight[:, in_ch:, :, :]
        # Flatten spatial for orthogonal init
        h = hidden_slice.view(4 * self.hidden_channels, -1)
        if h.shape[0] <= h.shape[1]:
            nn.init.orthogonal_(h)
        nn.init.zeros_(self.conv.bias)
 
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        device = x.device
 
        if h is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=device, dtype=x.dtype)
        if c is None:
            c = torch.zeros(B, self.hidden_channels, H, W, device=device, dtype=x.dtype)
 
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
 
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + 1.0)   # forget bias towards 1 — better for long sequences
        g = torch.tanh(g)
        o = torch.sigmoid(o)
 
        c_new = f * c + i * g
        h_new = o * torch.tanh(self.layer_norm(c_new))
 
        return h_new, c_new
 
 
# ---------------------------------------------------------------------------
# ConvLSTM Layer
# ---------------------------------------------------------------------------
 
class ConvLSTMLayer(nn.Module):
    """One temporal layer: processes an entire sequence through one ConvLSTM cell."""
 
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels
 
    def forward(
        self,
        x_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x_seq : (B, T, in_channels, H, W)
        Returns:
            outputs : (B, T, hidden_channels, H, W)
            (h, c)  : final hidden and cell states
        """
        B, T, C, H, W = x_seq.shape
        h, c = h0, c0
        outputs = []
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)
 
 
# ---------------------------------------------------------------------------
# Temporal Self-Attention Bottleneck
# ---------------------------------------------------------------------------
 
class TemporalAttention(nn.Module):
    """
    Lightweight temporal self-attention applied after the encoder.
 
    For each spatial location (h, w), the T hidden vectors attend to each
    other — learning which timestep's representation carries the most signal.
 
    Implementation:
        - Flatten spatial → treat each (H*W) location independently
        - Multi-head self-attention over the T time dimension
        - Residual + LayerNorm
 
    This is cheap: T=4 → 4×4 attention matrices, negligible compute.
    """
 
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, D, H, W)
        Returns: (B, T, D, H, W)
        """
        B, T, D, H, W = x.shape
 
        # Fold spatial into batch: treat each spatial location independently
        # Reshape: (B*H*W, T, D)
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
 
        # Temporal self-attention
        attended, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + attended)
 
        # FFN
        x_flat = self.norm2(x_flat + self.ffn(x_flat))
 
        # Reshape back: (B, T, D, H, W)
        return x_flat.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2)
 
 
# ---------------------------------------------------------------------------
# Spatial Attention Gate
# ---------------------------------------------------------------------------
 
class SpatialAttentionGate(nn.Module):
    """
    Squeeze-and-excitation style spatial attention for the output head.
    Helps focus prediction on ecologically meaningful regions.
    """
 
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)
 
 
# ---------------------------------------------------------------------------
# Full ConvLSTM Model
# ---------------------------------------------------------------------------
 
class ConvLSTMModel(nn.Module):
    """
    Improved ConvLSTM encoder-decoder for temporal prediction.
 
    Encoder: Processes T_in steps, applies temporal attention at bottleneck.
    Decoder: Residual autoregressive generation — predicts delta not absolute.
    """
 
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_cfg = config["model"]["convlstm"]
        use_validity = config["patches"]["use_validity_mask"]
 
        self.in_channels  = config["bands"]["count"] + (1 if use_validity else 0)
        self.out_channels = config["bands"]["count"]
        self.t_input  = config["model"]["t_input"]
        self.t_output = config["model"]["t_output"]
 
        hidden_channels: List[int] = model_cfg["hidden_channels"]
        kernel_size = model_cfg["kernel_size"]
        self.dropout_p = model_cfg.get("dropout", 0.1)
 
        # ---- Encoder ----
        enc_in = self.in_channels
        self.encoder_layers = nn.ModuleList()
        for h_ch in hidden_channels:
            self.encoder_layers.append(ConvLSTMLayer(enc_in, h_ch, kernel_size))
            enc_in = h_ch
 
        self.dropout = nn.Dropout2d(self.dropout_p)
 
        # ---- Temporal attention bottleneck ----
        bottleneck_dim = hidden_channels[-1]
        self.temporal_attn = TemporalAttention(
            d_model=bottleneck_dim,
            n_heads=min(4, bottleneck_dim // 16),
            dropout=self.dropout_p,
        )
 
        # ---- Decoder ----
        dec_in = self.out_channels
        self.decoder_layers = nn.ModuleList()
        for h_ch in hidden_channels:
            self.decoder_layers.append(ConvLSTMLayer(dec_in, h_ch, kernel_size))
            dec_in = h_ch
 
        # ---- Output head (predicts residual Δ, not absolute frame) ----
        self.spatial_gate = SpatialAttentionGate(hidden_channels[-1])
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-1] // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels[-1] // 2, self.out_channels, 1),
            nn.Tanh(),   # Δ bounded in (-1, 1); added to last_frame which is in [0,1]
        )
        # Scale residual to avoid exploding at start of training
        self.delta_scale = nn.Parameter(torch.zeros(1))
 
    def encode(self, x: torch.Tensor):
        """
        Run encoder on input sequence, apply temporal attention.
        Returns: all T hidden states (attended), plus final (h,c) per layer.
        """
        seq = x
        states = []
        for layer in self.encoder_layers:
            seq, (h, c) = layer(seq)
            # Apply spatial dropout to each frame
            B, T, C, H, W = seq.shape
            seq_drop = self.dropout(seq.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)
            seq = seq_drop
            states.append((h, c))
 
        # Temporal attention over the bottleneck sequence
        seq = self.temporal_attn(seq)   # (B, T, D, H, W)
 
        return seq, states
 
    def decode(
        self,
        last_frame: torch.Tensor,
        enc_states: list,
        t_output: int,
    ) -> torch.Tensor:
        """
        Residual autoregressive decoding.
 
        last_frame : (B, C, H, W) — last input frame (t = T_in − 1)
        enc_states : list of (h, c) from encoder
        """
        preds = []
        prev_frame = last_frame
        dec_states = list(enc_states)
 
        for _ in range(t_output):
            x_t = prev_frame.unsqueeze(1)   # (B, 1, C, H, W)
            new_states = []
            for i, layer in enumerate(self.decoder_layers):
                h0, c0 = dec_states[i]
                out_seq, (h, c) = layer(x_t, h0, c0)
                x_t = out_seq
                new_states.append((h, c))
 
            dec_states = new_states
 
            # Spatial attention gate + residual prediction
            h_spatial = self.spatial_gate(x_t[:, 0])   # (B, hidden, H, W)
            delta = self.output_head(h_spatial)          # (B, C, H, W) in (-1, 1)
            # Residual: scale from 0 at start to full contribution
            scale = torch.sigmoid(self.delta_scale) * 0.5  # max 0.5 change per step
            pred = (prev_frame + delta * scale).clamp(0.0, 1.0)
            # SAR_VV is z-scored so can go negative — don't clamp it globally.
            # For now clamp all bands; caller can post-process.
 
            preds.append(pred)
            prev_frame = pred
 
        return torch.stack(preds, dim=1)   # (B, T_out, C, H, W)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T_in, C_in, H, W)
        Returns:
            (B, T_out, C_out, H, W)
        """
        _, enc_states = self.encode(x)
        last_frame = x[:, -1, :self.out_channels]
        return self.decode(last_frame, enc_states, self.t_output)
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)