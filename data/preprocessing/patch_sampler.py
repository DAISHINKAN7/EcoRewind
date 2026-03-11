"""
patch_sampler.py
----------------
Extracts 128×128 spatial patches from (T, C, H, W) tensors using
a sliding window with configurable stride and NaN filtering.
 
Patches are saved as memory-mapped .npy arrays to keep RAM usage low
during DataLoader iteration on GPU instances.
 
Strategy:
  - Slide over H×W with stride (default 64 = 50% overlap)
  - Skip patches where NaN% > nan_threshold (default 10%)
  - Optionally append validity channel (8th channel)
  - Save patches indexed by (ecosystem, t_start, row, col)
"""
 
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
 
logger = logging.getLogger(__name__)
 
 
class PatchSampler:
    """
    Samples 128×128 patches from a time-series tensor.
 
    Each patch covers the full time axis T, so one patch has shape
    (T, C, H_patch, W_patch) = (32, 7, 128, 128).
    """
 
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patch_size = config["patches"]["size"]         # 128
        self.stride = config["patches"]["stride"]           # 64
        self.nan_threshold = config["patches"]["nan_threshold"]  # 0.10
        self.use_validity_mask = config["patches"]["use_validity_mask"]
        self.processed_dir = Path(config["data"]["processed_dir"])
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def build_patches(
        self,
        ecosystem: str,
        tensor: np.ndarray,
        validity: np.ndarray,
        normalizer,
        force_rebuild: bool = False,
    ) -> str:
        """
        Extract all valid patches and save to disk.
 
        Args:
            ecosystem  : "everglades" or "mississippi"
            tensor     : (T, C, H, W) normalized tensor
            validity   : (T, 1, H, W) binary validity mask
            normalizer : fitted EcoNormalizer (for inverse_transform in metrics)
            force_rebuild : overwrite existing patches
 
        Returns:
            path to saved patches directory
        """
        out_dir = self.processed_dir / ecosystem / "patches"
        index_path = out_dir / "patch_index.npy"
 
        if index_path.exists() and not force_rebuild:
            logger.info(
                f"[{ecosystem}] Patches already exist at {out_dir}. "
                "Pass force_rebuild=True to re-extract."
            )
            return str(out_dir)
 
        out_dir.mkdir(parents=True, exist_ok=True)
 
        T, C, H, W = tensor.shape
        p = self.patch_size
        s = self.stride
 
        # Add validity channel if requested
        if self.use_validity_mask:
            # validity shape: (T, 1, H, W) → take mean over time → (1, H, W) per patch
            pass  # handled per-patch below
 
        # Generate all candidate patch positions
        row_starts = list(range(0, H - p + 1, s))
        col_starts = list(range(0, W - p + 1, s))
 
        # Also include the last row/col to ensure edge coverage
        if row_starts and row_starts[-1] != H - p:
            row_starts.append(H - p)
        if col_starts and col_starts[-1] != W - p:
            col_starts.append(W - p)
 
        logger.info(
            f"[{ecosystem}] Sampling patches: "
            f"{len(row_starts)}×{len(col_starts)} = {len(row_starts)*len(col_starts)} candidates"
        )
 
        # Save patches and build index
        patch_index = []    # list of (row, col, nan_pct) for accepted patches
        n_saved = 0
        n_rejected = 0
 
        patches_file = out_dir / "patches.npy"
 
        # Two-pass: first count accepted, then allocate and fill
        accepted_positions = []
        for r in row_starts:
            for c_pos in col_starts:
                patch_data = tensor[:, :, r:r+p, c_pos:c_pos+p]   # (T, C, p, p)
                nan_pct = np.mean(~np.isfinite(patch_data))
                if nan_pct <= self.nan_threshold:
                    accepted_positions.append((r, c_pos, float(nan_pct)))
 
        n_accepted = len(accepted_positions)
        n_rejected = len(row_starts) * len(col_starts) - n_accepted
        logger.info(
            f"[{ecosystem}] Accepted: {n_accepted} patches | "
            f"Rejected: {n_rejected} (NaN > {self.nan_threshold*100:.0f}%)"
        )
 
        if n_accepted == 0:
            logger.error(
                f"[{ecosystem}] NO valid patches found! "
                f"Check NaN% in tensor and nan_threshold in config."
            )
            return str(out_dir)
 
        # Determine output channels
        out_C = C + 1 if self.use_validity_mask else C
 
        # Allocate memory-mapped array: (N, T, C_out, p, p)
        patches_arr = np.lib.format.open_memmap(
            str(patches_file),
            mode="w+",
            dtype=np.float32,
            shape=(n_accepted, T, out_C, p, p),
        )
 
        for i, (r, c_pos, nan_pct) in enumerate(accepted_positions):
            patch = tensor[:, :, r:r+p, c_pos:c_pos+p].astype(np.float32)  # (T, C, p, p)
 
            if self.use_validity_mask:
                # Validity channel: (T, 1, p, p)
                val_patch = validity[:, :, r:r+p, c_pos:c_pos+p].astype(np.float32)
                patch = np.concatenate([patch, val_patch], axis=1)  # (T, C+1, p, p)
 
            # Replace NaN with 0 for model input (masked by validity channel)
            patch_filled = np.where(np.isfinite(patch), patch, 0.0)
            patches_arr[i] = patch_filled
 
        del patches_arr  # flush memmap
 
        # Save index: (N, 3) array with [row, col, nan_pct]
        index_arr = np.array(accepted_positions, dtype=np.float32)
        np.save(str(index_path), index_arr)
 
        logger.info(
            f"[{ecosystem}] Saved {n_accepted} patches → {patches_file}\n"
            f"  Shape: ({n_accepted}, {T}, {out_C}, {p}, {p})\n"
            f"  Size:  {n_accepted * T * out_C * p * p * 4 / 1e9:.2f} GB"
        )
        return str(out_dir)
 
    # ------------------------------------------------------------------
    # Window splitting for train/val split
    # ------------------------------------------------------------------
 
    def get_windows(
        self,
        n_patches: int,
        T: int,
        t_event: int,
        t_input: int,
        t_output: int,
    mode: str = "pre_only",
    ) -> List[Tuple[int, int]]:
        """
        Generate all valid (t_start, t_end) sliding windows.
 
        mode="pre_only":
            Windows restricted to pre-disturbance period [0, t_event).
            Both input AND target must fall entirely before t_event.
            Gives Everglades only 1 window when t_event=6, t_input=4, t_output=2.
 
        mode="all":
            Windows slide across the FULL time series [0, T).
            Any window that fits within T is valid — no restriction on t_event.
            Gives Everglades ~28 windows instead of 1.
 
            Scientific justification: the model learns general vegetation dynamics.
            Post-event recovery data is valid training signal — the model sees the
            full state space (healthy → damaged → recovering). Counterfactual
            generation always uses pre-event context at inference time regardless.
 
            Only excluded: windows where t_event falls INSIDE the input slice
            (the model would be asked to predict from a mid-hurricane context,
            which is ambiguous). Those windows are skipped.
 
        Returns list of (t_start, t_start + t_input + t_output) tuples.
        """
        windows = []
        window_len = t_input + t_output
 
        if mode == "pre_only":
            for t_start in range(0, t_event - window_len + 1):
                windows.append((t_start, t_start + window_len))
 
        elif mode == "all":
            for t_start in range(0, T - window_len + 1):
                t_end = t_start + window_len
                # Skip windows where t_event falls inside the INPUT slice.
                # The input slice is [t_start, t_start + t_input).
                # If t_event is inside the input, the model gets a "contaminated"
                # context that spans the hurricane — ambiguous and undesirable.
                input_contains_event = (t_start <= t_event < t_start + t_input)
                if not input_contains_event:
                    windows.append((t_start, t_end))
        else:
            raise ValueError(f"Unknown window mode: '{mode}'. Use 'pre_only' or 'all'.")
 
        logger.info(
            f"Generated {len(windows)} sliding windows "
            f"(mode={mode}, t_input={t_input}, t_output={t_output}, "
            f"t_event={t_event}, T={T})"
        )
        
        return windows