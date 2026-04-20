"""
patch_sampler.py  (v2 — optical-only validity mask)
-----------------------------------------------------
KEY FIX: The 8th channel (validity mask) now reflects OPTICAL BAND validity only,
not ALL-band validity.

ORIGINAL BUG:
    valid_pixels = np.all(np.isfinite(data), axis=0, keepdims=True)
    # SAR is 87.5% NaN → validity is 87.5% zero
    # Loss computed on only 12.5% of pixels → catastrophic underfitting

FIX:
    optical_valid = np.all(np.isfinite(data[:6]), axis=0, keepdims=True)
    # Uses only optical bands (Blue/Green/Red/NIR/NDVI/NDWI)
    # These are 9.5% NaN → validity is 90.5% valid
    # Loss computed on 90.5% of pixels → proper training signal

SAR handling: SAR is excluded from the validity mask. The loss function
already down-weights SAR (weight=0.1). With only 12.5% valid SAR pixels
the SAR loss contribution is negligible regardless. Excluding it from
validity prevents it from contaminating the optical band training.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class PatchSampler:
    """
    Samples 128×128 patches from a time-series tensor.
    Each patch covers the full time axis T: shape (T, C+1, H, W)
    where the last channel is the OPTICAL validity mask.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patch_size = config["patches"]["size"]
        self.stride = config["patches"]["stride"]
        self.nan_threshold = config["patches"]["nan_threshold"]
        self.use_validity_mask = config["patches"]["use_validity_mask"]
        self.processed_dir = Path(config["data"]["processed_dir"])
        # SAR is the last band — exclude from optical validity
        self.sar_idx = config["bands"]["indices"].get("sar_vv", 6)
        self.n_bands = config["bands"]["count"]

    def build_patches(
        self,
        ecosystem: str,
        tensor: np.ndarray,
        validity: np.ndarray,
        normalizer,
        force_rebuild: bool = False,
    ) -> str:
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

        row_starts = list(range(0, H - p + 1, s))
        col_starts = list(range(0, W - p + 1, s))

        if row_starts and row_starts[-1] != H - p:
            row_starts.append(H - p)
        if col_starts and col_starts[-1] != W - p:
            col_starts.append(W - p)

        logger.info(
            f"[{ecosystem}] Sampling patches: "
            f"{len(row_starts)}×{len(col_starts)} = {len(row_starts)*len(col_starts)} candidates"
        )

        patches_file = out_dir / "patches.npy"

        # FIX: Filter on OPTICAL bands only (exclude SAR)
        # SAR is 87.5% NaN — using it for filtering rejects all Mississippi patches
        n_optical = min(self.sar_idx, C)   # bands 0..sar_idx-1

        accepted_positions = []
        for r in row_starts:
            for c_pos in col_starts:
                patch_data = tensor[:, :, r:r+p, c_pos:c_pos+p]
                optical_data = patch_data[:, :n_optical]
                nan_pct_optical = np.mean(~np.isfinite(optical_data))
                nan_pct_all = np.mean(~np.isfinite(patch_data))
                if nan_pct_optical <= self.nan_threshold:
                    accepted_positions.append((r, c_pos, float(nan_pct_all)))

        n_accepted = len(accepted_positions)
        n_rejected = len(row_starts) * len(col_starts) - n_accepted
        logger.info(
            f"[{ecosystem}] Accepted: {n_accepted} patches | "
            f"Rejected: {n_rejected} (optical NaN > {self.nan_threshold*100:.0f}%)"
        )

        if n_accepted == 0:
            logger.error(
                f"[{ecosystem}] NO valid patches found! "
                f"Check NaN% in tensor and nan_threshold in config."
            )
            return str(out_dir)

        # FIX: Output channels = bands + 1 validity (optical only)
        out_C = C + 1 if self.use_validity_mask else C

        patches_arr = np.lib.format.open_memmap(
            str(patches_file),
            mode="w+",
            dtype=np.float32,
            shape=(n_accepted, T, out_C, p, p),
        )

        for i, (r, c_pos, nan_pct) in enumerate(accepted_positions):
            patch = tensor[:, :, r:r+p, c_pos:c_pos+p].astype(np.float32)

            if self.use_validity_mask:
                # FIX: Use OPTICAL-ONLY validity (not all-band validity)
                # A pixel is "valid for training" if its optical bands are finite.
                # SAR being NaN at the same pixel doesn't disqualify the optical data.
                optical_patch = patch[:, :n_optical]   # (T, n_optical, p, p)
                optical_valid = np.all(np.isfinite(optical_patch), axis=1, keepdims=True)
                # Shape: (T, 1, p, p), 1=optically valid, 0=cloud/nodata
                patch = np.concatenate([patch, optical_valid.astype(np.float32)], axis=1)

            # Fill NaN with CORRECT per-band neutral values (not universal 0.0)
            # FIX: Use band-specific fills so model doesn't learn wrong mean
            band_names = self.config["bands"]["names"]
            norm_methods = self.config["bands"]["normalization"]
            for c in range(min(C, patch.shape[1])):
                if c >= len(band_names):
                    break
                name = band_names[c].lower()
                method = norm_methods.get(name, "minmax")
                # 0.5 for [0,1]-space bands (neutral mid-point, not "dead" NDVI=-1)
                # 0.0 for z-scored bands (correct: z-scored mean IS 0)
                fill_val = 0.5 if method in ("minmax", "shift") else 0.0
                band = patch[:, c]
                patch[:, c] = np.where(np.isfinite(band), band, fill_val)

            # Validity channel: already binary, fill any NaN with 0 (invalid)
            if self.use_validity_mask:
                val_ch = patch[:, C:]
                patch[:, C:] = np.where(np.isfinite(val_ch), val_ch, 0.0)

            patches_arr[i] = patch

        del patches_arr  # flush memmap

        index_arr = np.array(accepted_positions, dtype=np.float32)
        np.save(str(index_path), index_arr)

        logger.info(
            f"[{ecosystem}] Saved {n_accepted} patches → {patches_file}\n"
            f"  Shape: ({n_accepted}, {T}, {out_C}, {p}, {p})\n"
            f"  Size:  {n_accepted * T * out_C * p * p * 4 / 1e9:.2f} GB\n"
            f"  NOTE: Validity channel = OPTICAL bands only (SAR excluded)"
        )
        return str(out_dir)

    def get_windows(
        self,
        n_patches: int,
        T: int,
        t_event: int,
        t_input: int,
        t_output: int,
        mode: str = "pre_only",
    ) -> List[Tuple[int, int]]:
        windows = []
        window_len = t_input + t_output

        if mode == "pre_only":
            for t_start in range(0, t_event - window_len + 1):
                windows.append((t_start, t_start + window_len))
        elif mode == "all":
            for t_start in range(0, T - window_len + 1):
                t_end = t_start + window_len
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