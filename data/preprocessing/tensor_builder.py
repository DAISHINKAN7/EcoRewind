"""
tensor_builder.py
-----------------
Stacks all quarterly composites for an ecosystem into a single
(T, C, H, W) float32 numpy array and saves it to disk.
 
Also builds a companion validity tensor (T, 1, H, W) marking
valid pixels across time (for POOR quality composites).
 
Run via: python scripts/01_build_tensors.py
"""
 
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
 
from data.loaders.composite_loader import CompositeLoader, EcosystemBundle
 
logger = logging.getLogger(__name__)
 
 
class TensorBuilder:
    """
    Builds (T, C, H, W) tensors from discovered composite files.
 
    For missing quarters (e.g. Mississippi v2 pending), fills with NaN
    placeholders so the time axis is always consistent (length T=32).
    """
 
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loader = CompositeLoader(config)
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.n_bands = config["bands"]["count"]
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def build(self, ecosystem: str, force_rebuild: bool = False) -> Dict[str, np.ndarray]:
        """
        Build tensors for one ecosystem. Returns dict with keys:
          "tensor"   → (T, C, H, W) full time-series
          "validity" → (T, 1, H, W) binary mask (1=valid pixel)
        Saves .npy files to processed/ folder.
        """
        out_dir = self.processed_dir / ecosystem.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
 
        tensor_path = out_dir / "tensor.npy"
        validity_path = out_dir / "validity.npy"
 
        if tensor_path.exists() and not force_rebuild:
            logger.info(f"[{ecosystem}] Tensor already exists. Loading from {tensor_path}")
            logger.info("  Pass force_rebuild=True to rebuild from TIFFs.")
            tensor = np.load(str(tensor_path))
            validity = np.load(str(validity_path))
            self._print_tensor_summary(ecosystem, tensor)
            return {"tensor": tensor, "validity": validity}
 
        # Load composite file list
        bundle = self.loader.load_ecosystem(ecosystem)
 
        # Determine output spatial dimensions from first readable file
        H, W = self._probe_dimensions(bundle)
        eco_cfg = self.config["ecosystems"][ecosystem.lower()]
        T = eco_cfg["n_composites"]
 
        logger.info(f"[{ecosystem}] Allocating tensor ({T}, {self.n_bands}, {H}, {W}) ...")
        tensor = np.full((T, self.n_bands, H, W), np.nan, dtype=np.float32)
        validity = np.zeros((T, 1, H, W), dtype=np.float32)
 
        # Build mask lookup: (year, quarter) → mask array
        mask_lookup = self._build_mask_lookup(bundle, H, W)
 
        # Fill tensor
        found_indices = set()
        for cf in bundle.composites:
            t = cf.time_index
            if t < 0 or t >= T:
                logger.warning(f"  Skipping {cf.label}: time_index {t} out of range [0, {T})")
                continue
 
            logger.info(f"  [{t:02d}/{T-1}] Reading {cf.label} ({cf.quality}) ...")
            try:
                data = self.loader.read_composite(cf)
            except Exception as e:
                logger.error(f"  Failed to read {cf.path}: {e}")
                continue
 
            # Resize if needed (minor dimension differences across composites)
            data = self._resize_if_needed(data, H, W, cf.label)
 
            # Check band count
            if data.shape[0] != self.n_bands:
                logger.warning(
                    f"  {cf.label}: expected {self.n_bands} bands, got {data.shape[0]}. "
                    f"Padding with NaN."
                )
                pad = np.full((self.n_bands - data.shape[0], H, W), np.nan, dtype=np.float32)
                data = np.concatenate([data, pad], axis=0)
 
            tensor[t] = data
            found_indices.add(t)
 
            # Validity mask: 1 where all bands are finite
            valid_pixels = np.all(np.isfinite(data), axis=0, keepdims=True)
 
            # Override with pre-computed mask file if available
            key = (cf.year, cf.quarter)
            if key in mask_lookup:
                validity[t] = mask_lookup[key]
                logger.info(f"    Using pre-computed validity mask for {cf.label}")
            else:
                validity[t] = valid_pixels.astype(np.float32)
 
        # Report missing quarters
        all_indices = set(range(T))
        missing = sorted(all_indices - found_indices)
        if missing:
            logger.warning(
                f"[{ecosystem}] {len(missing)} quarters missing (NaN placeholders):\n"
                f"  Indices: {missing}\n"
                f"  This is expected for Mississippi v2 pending re-export."
            )
 
        # Validate disturbance signal
        self._validate_disturbance(ecosystem, tensor)
 
        # Save
        np.save(str(tensor_path), tensor)
        np.save(str(validity_path), validity)
        logger.info(f"[{ecosystem}] Saved tensor → {tensor_path}")
        logger.info(f"[{ecosystem}] Saved validity → {validity_path}")
 
        self._print_tensor_summary(ecosystem, tensor)
        return {"tensor": tensor, "validity": validity}
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _probe_dimensions(self, bundle: EcosystemBundle) -> Tuple[int, int]:
        """Read the first composite to determine spatial dimensions."""
        for cf in bundle.composites:
            try:
                import rasterio
                with rasterio.open(cf.path) as src:
                    return src.height, src.width
            except Exception:
                continue
        raise RuntimeError(f"Could not determine dimensions from any file in {bundle.ecosystem}")
 
    def _build_mask_lookup(
        self, bundle: EcosystemBundle, H: int, W: int
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Load pre-computed validity mask files into a lookup dict."""
        lookup = {}
        for mf in bundle.masks:
            try:
                mask_data = self.loader.read_tif(mf.path)
                # Mask file should be (1, H, W) or (H, W)
                if mask_data.ndim == 2:
                    mask_data = mask_data[np.newaxis]
                mask_data = self._resize_if_needed(mask_data, H, W, mf.label)
                lookup[(mf.year, mf.quarter)] = mask_data[[0]].astype(np.float32)
                logger.info(f"  Loaded mask: {mf.label}")
            except Exception as e:
                logger.warning(f"  Could not load mask {mf.path}: {e}")
        return lookup
 
    def _resize_if_needed(
        self, data: np.ndarray, target_H: int, target_W: int, label: str
    ) -> np.ndarray:
        """Crop or pad data to (C, target_H, target_W) if dimensions mismatch."""
        C, H, W = data.shape
        if H == target_H and W == target_W:
            return data
 
        logger.debug(
            f"  {label}: spatial mismatch ({H}×{W}) vs target ({target_H}×{target_W}) — cropping"
        )
        # Crop to minimum overlap, then pad
        h_crop = min(H, target_H)
        w_crop = min(W, target_W)
        out = np.full((C, target_H, target_W), np.nan, dtype=np.float32)
        out[:, :h_crop, :w_crop] = data[:, :h_crop, :w_crop]
        return out
 
    def _validate_disturbance(self, ecosystem: str, tensor: np.ndarray) -> None:
        """
        Quick sanity check: confirm hurricane signal is present.
        Everglades: NDVI drop at t_event vs baseline.
        Mississippi: warn if NDVI is near zero (v1 wrong AOI).
        """
        eco_cfg = self.config["ecosystems"][ecosystem.lower()]
        t_event = eco_cfg["t_event"]
        ndvi_idx = self.config["bands"]["indices"]["ndvi"]
 
        ndvi_series = tensor[:, ndvi_idx]  # (T, H, W)
 
        def mean_ndvi(t: int) -> Optional[float]:
            if t < 0 or t >= tensor.shape[0]:
                return None
            vals = ndvi_series[t]
            finite = vals[np.isfinite(vals)]
            return float(np.mean(finite)) if len(finite) > 0 else None
 
        logger.info(f"\n[{ecosystem.upper()}] Disturbance signal validation:")
 
        if ecosystem.lower() == "everglades":
            # Baseline: average of same-quarter (Q3) in prior 2 years
            baseline_t = [t for t in [t_event - 4, t_event - 8] if t >= 0]
            baseline_ndvi = [mean_ndvi(t) for t in baseline_t if mean_ndvi(t) is not None]
            event_ndvi = mean_ndvi(t_event)
 
            if baseline_ndvi and event_ndvi is not None:
                baseline_mean = np.mean(baseline_ndvi)
                delta = event_ndvi - baseline_mean
                logger.info(f"  Baseline NDVI (Q3 prior years): {baseline_mean:.4f}")
                logger.info(f"  Event NDVI (t={t_event}):        {event_ndvi:.4f}")
                logger.info(f"  ΔNDVI at impact:                 {delta:.4f}")
                if delta < -0.05:
                    logger.info("  ✅ DROP CONFIRMED — hurricane signal present")
                else:
                    logger.warning(
                        f"  ⚠️  Weak drop ({delta:.4f}). Expected ~−0.113 for Irma.\n"
                        f"  Check t_event index in configs/config.yaml."
                    )
            else:
                logger.warning("  Could not compute baseline NDVI (missing data)")
 
        elif ecosystem.lower() == "mississippi":
            threshold = eco_cfg.get("ndvi_mean_threshold", 0.10)
            overall_ndvi = [mean_ndvi(t) for t in range(tensor.shape[0])]
            overall_ndvi = [v for v in overall_ndvi if v is not None]
            if overall_ndvi:
                mean_overall = np.mean(overall_ndvi)
                logger.info(f"  Mean NDVI across all timesteps: {mean_overall:.4f}")
                if mean_overall < threshold:
                    logger.warning(
                        f"  ⚠️  Very low NDVI ({mean_overall:.4f} < {threshold}).\n"
                        f"  This matches the v1 open-water delta AOI (wrong location).\n"
                        f"  Hurricane Ida signal will NOT be detectable.\n"
                        f"  ACTION: Re-export from Barataria Bay (29.45°N, −90.7°W)\n"
                        f"  and set mississippi.version='v2' in config.yaml."
                    )
                else:
                    logger.info(
                        f"  ✅ NDVI {mean_overall:.4f} ≥ threshold {threshold} — "
                        f"looks like vegetated marsh (v2)"
                    )
 
    def _print_tensor_summary(self, ecosystem: str, tensor: np.ndarray) -> None:
        T, C, H, W = tensor.shape
        band_names = self.config["bands"]["names"]
        logger.info(f"\n[{ecosystem.upper()}] Tensor summary:")
        logger.info(f"  Shape: (T={T}, C={C}, H={H}, W={W})")
        logger.info(f"  Size:  {tensor.nbytes / 1e9:.2f} GB")
        logger.info(f"  {'Band':>10}  {'Mean':>8}  {'Std':>8}  {'NaN%':>6}")
        logger.info("  " + "-" * 38)
        for c in range(C):
            band = tensor[:, c]
            finite = band[np.isfinite(band)]
            nan_pct = 100 * (1 - len(finite) / band.size)
            mean_val = float(np.mean(finite)) if len(finite) > 0 else float("nan")
            std_val = float(np.std(finite)) if len(finite) > 0 else float("nan")
            name = band_names[c] if c < len(band_names) else f"Band{c}"
            logger.info(f"  {name:>10}  {mean_val:>8.4f}  {std_val:>8.4f}  {nan_pct:>5.1f}%")
        logger.info("")