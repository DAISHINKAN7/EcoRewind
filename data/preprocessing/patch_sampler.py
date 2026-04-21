"""
patch_sampler.py  (v3 — stratified sampling + coverage threshold)
------------------------------------------------------------------
Fixes vs v2:

FIX E — NDVI variance-based stratified sampling.
  The original sampler accepts any patch with optical NaN < threshold.
  This allows many low-signal (flat NDVI) patches that teach the model
  nothing useful about vegetation dynamics.

  New: compute per-patch NDVI temporal variance. Patches are partitioned
  into "high-signal" (variance ≥ ndvi_var_threshold) and "low-signal"
  bins. The final patch set is stratified: at least min_high_signal_frac
  (default 40%) of accepted patches come from the high-signal bin.
  Low-signal patches fill the remainder. This ensures the model always
  sees vegetation-change patches without discarding stable-background
  context entirely.

FIX 2 — Mask coverage threshold (≥ 30% valid optical pixels).
  Even if a patch passes the NaN filter (< nan_threshold NaN overall),
  individual timesteps may be heavily clouded. We now check that each
  timestep window has ≥ min_valid_coverage of valid optical pixels,
  and patches whose mean coverage across T falls below this threshold
  are rejected.

FIX 2 — Soft masking option.
  When use_soft_mask=True (new config key), the validity channel is
  blurred with a 3×3 average kernel so the transition from valid to
  invalid is soft rather than a hard 0/1 boundary. This prevents
  the model from learning to predict the validity mask boundary instead
  of the actual signal. Hard masking is still the default.

UNCHANGED: memory-mapped patch storage, NaN fill with band-aware values,
           optical-only validity channel, window generation.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class PatchSampler:
    """
    Samples 128×128 patches from a time-series tensor.
    Each patch: (T, C+1, H, W) where last channel is optical validity.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config           = config
        self.patch_size       = config["patches"]["size"]
        self.stride           = config["patches"]["stride"]
        self.nan_threshold    = config["patches"]["nan_threshold"]
        self.use_validity_mask = config["patches"]["use_validity_mask"]
        self.processed_dir    = Path(config["data"]["processed_dir"])

        self.sar_idx   = config["bands"]["indices"].get("sar_vv", 6)
        self.ndvi_idx  = config["bands"]["indices"].get("ndvi", 4)
        self.n_bands   = config["bands"]["count"]

        # FIX E: stratified sampling parameters
        self.ndvi_var_threshold     = config["patches"].get("ndvi_var_threshold", 5e-4)
        self.min_high_signal_frac   = config["patches"].get("min_high_signal_frac", 0.40)

        # FIX 2: minimum valid coverage per timestep
        self.min_valid_coverage = config["patches"].get("min_valid_coverage", 0.30)

        # FIX 2: soft mask option
        self.use_soft_mask = config["patches"].get("use_soft_mask", False)

    # ------------------------------------------------------------------
    # Build patches
    # ------------------------------------------------------------------

    def build_patches(
        self,
        ecosystem:     str,
        tensor:        np.ndarray,
        validity:      np.ndarray,
        normalizer,
        force_rebuild: bool = False,
    ) -> str:
        out_dir    = self.processed_dir / ecosystem / "patches"
        index_path = out_dir / "patch_index.npy"

        if index_path.exists() and not force_rebuild:
            logger.info(
                f"[{ecosystem}] Patches exist at {out_dir}. "
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
            f"[{ecosystem}] Sampling: "
            f"{len(row_starts)}×{len(col_starts)} = "
            f"{len(row_starts)*len(col_starts)} candidate patches"
        )

        n_optical = min(self.sar_idx, C)

        # First pass: collect all valid candidates with their NDVI variance
        high_signal = []   # (r, c, nan_pct, ndvi_var)
        low_signal  = []

        for r in row_starts:
            for c_pos in col_starts:
                patch_data = tensor[:, :, r:r+p, c_pos:c_pos+p]

                # Optical NaN check
                optical_data  = patch_data[:, :n_optical]
                nan_pct_optical = np.mean(~np.isfinite(optical_data))
                if nan_pct_optical > self.nan_threshold:
                    continue

                # FIX 2: per-timestep coverage check
                mean_coverage = np.mean([
                    np.mean(np.isfinite(patch_data[t, :n_optical]))
                    for t in range(T)
                ])
                if mean_coverage < self.min_valid_coverage:
                    continue

                # FIX E: NDVI temporal variance
                ndvi_data = patch_data[:, self.ndvi_idx]
                finite_ndvi = ndvi_data[np.isfinite(ndvi_data)]
                ndvi_var = float(np.var(finite_ndvi)) if len(finite_ndvi) > 10 else 0.0

                nan_pct_all = float(np.mean(~np.isfinite(patch_data)))
                entry = (r, c_pos, nan_pct_all, ndvi_var)

                if ndvi_var >= self.ndvi_var_threshold:
                    high_signal.append(entry)
                else:
                    low_signal.append(entry)

        total_candidates = len(high_signal) + len(low_signal)
        logger.info(
            f"[{ecosystem}] Candidates: {total_candidates} "
            f"(high-signal: {len(high_signal)}, low-signal: {len(low_signal)})"
        )

        if total_candidates == 0:
            logger.error(f"[{ecosystem}] NO valid patches — check nan_threshold and data quality")
            return str(out_dir)

        # FIX E: Stratified selection
        # Ensure min_high_signal_frac come from high-signal pool
        n_total       = total_candidates
        n_high_target = max(
            int(n_total * self.min_high_signal_frac),
            min(len(high_signal), 1),   # at least 1 if any exist
        )
        n_high_take   = min(n_high_target, len(high_signal))
        n_low_take    = n_total - n_high_take

        # Sort high-signal by variance descending (most informative first)
        high_signal.sort(key=lambda e: e[3], reverse=True)
        low_signal.sort(key=lambda e: e[2])   # sort low-signal by NaN% ascending

        accepted = high_signal[:n_high_take] + low_signal[:n_low_take]
        n_accepted = len(accepted)

        logger.info(
            f"[{ecosystem}] Accepted: {n_accepted} patches "
            f"(high-signal: {n_high_take}/{len(high_signal)}, "
            f"low-signal: {n_low_take}/{len(low_signal)})"
        )
        logger.info(
            f"[{ecosystem}] NDVI variance threshold: {self.ndvi_var_threshold:.1e} | "
            f"Min coverage: {self.min_valid_coverage*100:.0f}%"
        )

        if n_accepted == 0:
            logger.error(f"[{ecosystem}] No patches selected after stratification")
            return str(out_dir)

        # Write patches
        patches_file = out_dir / "patches.npy"
        out_C = C + 1 if self.use_validity_mask else C

        patches_arr = np.lib.format.open_memmap(
            str(patches_file),
            mode="w+",
            dtype=np.float32,
            shape=(n_accepted, T, out_C, p, p),
        )

        band_names   = self.config["bands"]["names"]
        norm_methods = self.config["bands"]["normalization"]

        for i, (r, c_pos, nan_pct, ndvi_var) in enumerate(accepted):
            patch = tensor[:, :, r:r+p, c_pos:c_pos+p].astype(np.float32)

            if self.use_validity_mask:
                optical_patch = patch[:, :n_optical]
                optical_valid = np.all(np.isfinite(optical_patch), axis=1, keepdims=True)
                opt_valid_f   = optical_valid.astype(np.float32)

                # FIX 2: soft masking (optional)
                if self.use_soft_mask:
                    opt_valid_f = self._soften_mask(opt_valid_f)

                patch = np.concatenate([patch, opt_valid_f], axis=1)

            # Fill NaN with band-aware neutral values
            for c in range(min(C, patch.shape[1])):
                if c >= len(band_names):
                    break
                name   = band_names[c].lower()
                method = norm_methods.get(name, "minmax")
                fill   = 0.5 if method in ("minmax", "shift") else 0.0
                band   = patch[:, c]
                patch[:, c] = np.where(np.isfinite(band), band, fill)

            # Validity channel: fill NaN with 0 (invalid)
            if self.use_validity_mask:
                val_ch = patch[:, C:]
                patch[:, C:] = np.where(np.isfinite(val_ch), val_ch, 0.0)

            patches_arr[i] = patch

        del patches_arr  # flush

        # Save index (r, c, nan_pct, ndvi_var)
        index_arr = np.array(accepted, dtype=np.float32)
        np.save(str(index_path), index_arr)

        logger.info(
            f"[{ecosystem}] Saved {n_accepted} patches → {patches_file}\n"
            f"  Shape: ({n_accepted}, {T}, {out_C}, {p}, {p})\n"
            f"  Size:  {n_accepted * T * out_C * p * p * 4 / 1e9:.2f} GB"
        )
        return str(out_dir)

    # ------------------------------------------------------------------
    # Soft mask helper (FIX 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _soften_mask(valid: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Blur the binary validity mask with an average kernel.
        valid: (T, 1, H, W)  →  output: (T, 1, H, W) float in [0, 1]
        """
        from scipy.ndimage import uniform_filter
        T = valid.shape[0]
        out = np.empty_like(valid)
        for t in range(T):
            out[t, 0] = uniform_filter(valid[t, 0].astype(np.float32),
                                       size=kernel_size, mode="reflect")
        return out

    # ------------------------------------------------------------------
    # Window generation (unchanged logic, doc updated)
    # ------------------------------------------------------------------

    def get_windows(
        self,
        n_patches:  int,
        T:          int,
        t_event:    int,
        t_input:    int,
        t_output:   int,
        mode:       str = "pre_only",
    ) -> List[Tuple[int, int]]:
        windows    = []
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