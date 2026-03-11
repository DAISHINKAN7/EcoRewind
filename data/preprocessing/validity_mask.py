"""
validity_mask.py
----------------
Generates binary validity masks for POOR quality composites
(Florida summer quarters with 40–60% cloud cover NaN).
 
A validity mask is a single-band GeoTIFF (1=valid, 0=masked)
that can be pre-computed from existing composites and saved for reuse.
 
Also provides utilities for aggregating masks across time.
"""
 
import logging
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Any, Optional, List
 
logger = logging.getLogger(__name__)
 
 
def generate_validity_mask(
    tif_path: str,
    output_path: Optional[str] = None,
    nan_value: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a per-pixel validity mask from a composite GeoTIFF.
 
    A pixel is valid (1) if ALL bands are finite.
    A pixel is invalid (0) if ANY band is NaN or nodata.
 
    Args:
        tif_path    : path to source composite GeoTIFF
        output_path : if given, save mask as single-band GeoTIFF
        nan_value   : explicit nodata value to treat as invalid (auto-detected if None)
 
    Returns:
        (H, W) float32 array with values in {0, 1}
    """
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)    # (C, H, W)
        nodata = src.nodata if nan_value is None else nan_value
        profile = src.profile.copy()
 
        if nodata is not None:
            data[data == nodata] = np.nan
 
    # Pixel is valid only if all bands are finite
    valid = np.all(np.isfinite(data), axis=0).astype(np.float32)  # (H, W)
 
    nan_pct = 100.0 * (1.0 - np.mean(valid))
    logger.info(
        f"  Validity mask: {nan_pct:.1f}% invalid pixels "
        f"(valid: {np.sum(valid):.0f} / {valid.size})"
    )
 
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update({
            "count": 1,
            "dtype": "float32",
            "nodata": None,
        })
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(valid[np.newaxis])
        logger.info(f"  Saved validity mask → {output_path}")
 
    return valid
 
 
def generate_masks_for_poor_quarters(
    config: Dict[str, Any],
    ecosystem: str,
    composite_files: List,    # List[CompositeFile]
    force_rebuild: bool = False,
) -> Dict[str, str]:
    """
    Generate validity masks for all POOR quality composites in an ecosystem.
    Masks saved to: processed/<ecosystem>/masks/<label>_mask.npy
 
    Returns dict mapping label → mask_path.
    """
    eco_cfg = config["ecosystems"][ecosystem.lower()]
    poor_set = {
        (q["year"], q["quarter"])
        for q in eco_cfg.get("poor_quarters", [])
    }
    fair_set = {
        (q["year"], q["quarter"])
        for q in eco_cfg.get("fair_quarters", [])
    }
    target_set = poor_set | fair_set
 
    processed_dir = Path(config["data"]["processed_dir"])
    mask_dir = processed_dir / ecosystem.lower() / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
 
    mask_paths = {}
    for cf in composite_files:
        if (cf.year, cf.quarter) not in target_set:
            continue
 
        mask_path = mask_dir / f"{cf.label}_validity.npy"
 
        if mask_path.exists() and not force_rebuild:
            logger.info(f"  Mask exists: {cf.label}")
            mask_paths[cf.label] = str(mask_path)
            continue
 
        logger.info(f"  Generating mask for POOR quarter: {cf.label}")
        mask = generate_validity_mask(cf.path)
        np.save(str(mask_path), mask)
        mask_paths[cf.label] = str(mask_path)
 
    logger.info(
        f"[{ecosystem}] Generated {len(mask_paths)} validity masks → {mask_dir}"
    )
    return mask_paths
 
 
def aggregate_validity_over_time(validity_tensor: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel fraction of valid timesteps.
 
    Args:
        validity_tensor : (T, 1, H, W) binary validity tensor
 
    Returns:
        (H, W) float32 array with values in [0, 1]
        1.0 = valid in all timesteps, 0.5 = valid in 50% of timesteps
    """
    return np.mean(validity_tensor[:, 0], axis=0)   # (H, W)
 
 
def get_always_valid_mask(validity_tensor: np.ndarray) -> np.ndarray:
    """
    Returns a (H, W) mask where pixels are valid in ALL timesteps.
    Useful for restricting evaluation to spatially consistent areas.
    """
    return np.all(validity_tensor[:, 0] == 1.0, axis=0)
 
 
def apply_validity_to_tensor(
    tensor: np.ndarray, validity: np.ndarray, fill_value: float = 0.0
) -> np.ndarray:
    """
    Replace invalid pixels in tensor with fill_value.
 
    Args:
        tensor   : (T, C, H, W)
        validity : (T, 1, H, W) binary mask
        fill_value : value to substitute for invalid pixels
 
    Returns:
        (T, C, H, W) tensor with invalid pixels replaced
    """
    mask = validity == 0.0   # (T, 1, H, W), True where invalid
    mask_broadcast = np.broadcast_to(mask, tensor.shape)
    return np.where(mask_broadcast, fill_value, tensor)
 