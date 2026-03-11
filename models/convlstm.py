"""
convlstm.py
-----------
ConvLSTM baseline model for ECO-REWIND.
 
Architecture:
  Encoder: N stacked ConvLSTM cells process the input sequence
  Decoder: ConvLSTM unrolls T_out steps with a linear prediction head
 
Input:  (B, T_in, C, H, W)
Output: (B, T_out, C, H, W)
 
~5M parameters at default config.
Comfortable on RTX A4000 at batch=4, 128×128 patches.
"""
 
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
 
 
# ---------------------------------------------------------------------------
# ConvLSTM Cell
# ---------------------------------------------------------------------------
 
class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.
    Gates computed with spatial convolutions instead of linear transforms.
    """
 
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
 
        # All four gates in one convolution for efficiency
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        # Layer norm on hidden state for training stability
        self.layer_norm = nn.GroupNorm(
            num_groups=min(8, hidden_channels),
            num_channels=hidden_channels
        )
 
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, in_channels, H, W)
            h : (B, hidden_channels, H, W) or None
            c : (B, hidden_channels, H, W) or None
        Returns:
            h_new, c_new : same shape as h, c
        """
        B, _, H, W = x.shape
        device = x.device
 
        if h is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=device, dtype=x.dtype)
        if c is None:
            c = torch.zeros(B, self.hidden_channels, H, W, device=device, dtype=x.dtype)
 
        combined = torch.cat([x, h], dim=1)   # (B, in_ch + hidden, H, W)
        gates = self.conv(combined)             # (B, 4*hidden, H, W)
 
        # Split into 4 gates
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)    # input gate
        f = torch.sigmoid(f)    # forget gate
        g = torch.tanh(g)       # cell gate
        o = torch.sigmoid(o)    # output gate
 
        c_new = f * c + i * g
        h_new = o * torch.tanh(self.layer_norm(c_new))
 
        return h_new, c_new
 
 
# ---------------------------------------------------------------------------
# ConvLSTM Layer (sequence of cells)
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
# Full ConvLSTM Model
# ---------------------------------------------------------------------------
 
class ConvLSTMModel(nn.Module):
    """
    Multi-layer ConvLSTM encoder-decoder for temporal prediction.
 
    Encoder: Processes T_in steps, keeps final hidden state.
    Decoder: Autoregressively generates T_out steps.
    """
 
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_cfg = config["model"]["convlstm"]
        use_validity = config["patches"]["use_validity_mask"]
 
        # Input channels: C bands + optional validity mask channel
        self.in_channels = config["bands"]["count"] + (1 if use_validity else 0)
        self.out_channels = config["bands"]["count"]  # always predict C bands, not mask
        self.t_input = config["model"]["t_input"]
        self.t_output = config["model"]["t_output"]
 
        hidden_channels = model_cfg["hidden_channels"]     # e.g. [64, 64, 64]
        kernel_size = model_cfg["kernel_size"]
        self.dropout_p = model_cfg.get("dropout", 0.1)
 
        # Build encoder layers
        enc_in = self.in_channels
        self.encoder_layers = nn.ModuleList()
        for h_ch in hidden_channels:
            self.encoder_layers.append(ConvLSTMLayer(enc_in, h_ch, kernel_size))
            enc_in = h_ch
 
        self.dropout = nn.Dropout2d(self.dropout_p)
 
        # Decoder uses same hidden_channels (shares architecture, not weights)
        dec_in = self.out_channels  # decoder input = previous prediction
        self.decoder_layers = nn.ModuleList()
        for h_ch in hidden_channels:
            self.decoder_layers.append(ConvLSTMLayer(dec_in, h_ch, kernel_size))
            dec_in = h_ch
 
        # Final 1×1 conv: maps last hidden to output bands
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-1] // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[-1] // 2, self.out_channels, 1),
        )
 
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    def encode(self, x: torch.Tensor):
        """
        Run encoder on input sequence.
        Returns list of (h, c) tuples for each layer.
        """
        seq = x
        states = []
        for layer in self.encoder_layers:
            seq, (h, c) = layer(seq)
            seq = self.dropout(seq.view(-1, seq.shape[2], seq.shape[3], seq.shape[4])).view(seq.shape)
            states.append((h, c))
        return seq, states
 
    def decode(self, last_frame: torch.Tensor, enc_states, t_output: int) -> torch.Tensor:
        """
        Autoregressively decode T_out steps.
 
        last_frame : (B, C, H, W) — last input frame (t = T_in − 1)
        enc_states : list of (h, c) from encoder
        """
        preds = []
        prev_frame = last_frame      # (B, C, H, W)
        dec_states = list(enc_states)
 
        for _ in range(t_output):
            # Run each decoder layer for one step
            x_t = prev_frame.unsqueeze(1)   # (B, 1, C, H, W)
            new_states = []
            for i, layer in enumerate(self.decoder_layers):
                h0, c0 = dec_states[i]
                out_seq, (h, c) = layer(x_t, h0, c0)
                x_t = out_seq                # (B, 1, hidden, H, W)
                new_states.append((h, c))
 
            dec_states = new_states
            # Map hidden to output
            pred = self.output_head(x_t[:, 0])    # (B, out_channels, H, W)
            preds.append(pred)
            # Feed prediction back — strip validity channel if present
            prev_frame = pred
 
        return torch.stack(preds, dim=1)   # (B, T_out, C, H, W)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T_in, C, H, W)  — normalized input sequence
        Returns:
            (B, T_out, C_out, H, W) — predicted sequence
        """
        _, enc_states = self.encode(x)
        last_frame = x[:, -1, :self.out_channels]   # (B, C_out, H, W)
        return self.decode(last_frame, enc_states, self.t_output)
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)