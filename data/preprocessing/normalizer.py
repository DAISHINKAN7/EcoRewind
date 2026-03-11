"""
normalizer.py
-------------
Fits all normalizers exclusively on pre-disturbance data (t < t_event)
to prevent data leakage from hurricane-affected values.
 
Three methods:
  - minmax  : [0, 1] scaling (reflectance bands)
  - shift   : add 1, divide 2 → [0, 1] (NDVI, NDWI which are in [−1, 1])
  - zscore  : subtract mean, divide std (SAR_VV in dB)
 
Saves/loads normalizer stats as JSON for reproducibility.
"""
 
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
 
logger = logging.getLogger(__name__)
 
 
class EcoNormalizer:
    """
    Fits and applies per-band normalization.
 
    CRITICAL: Always call fit() with pre-disturbance slices only.
    Applying a normalizer fitted on the full time-series leaks
    post-hurricane values into the model's input statistics.
    """
 
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.band_names = config["bands"]["names"]
        self.norm_methods = config["bands"]["normalization"]
        self.n_bands = config["bands"]["count"]
        self._stats: Dict[str, Dict[str, float]] = {}
        self._fitted = False
 
    # ------------------------------------------------------------------
    # Fitting (pre-disturbance only)
    # ------------------------------------------------------------------
 
    def fit(self, tensor: np.ndarray, t_event: int) -> None:
        """
        Compute normalization statistics from pre-disturbance slice only.
 
        Args:
            tensor   : (T, C, H, W) full time-series
            t_event  : index of the hurricane quarter — fitting uses [0, t_event)
        """
        pre_tensor = tensor[:t_event]  # shape (t_event, C, H, W)
        logger.info(
            f"Fitting normalizer on t=0..{t_event-1} ({t_event} pre-disturbance quarters)"
        )
 
        self._stats = {}
        for c, name in enumerate(self.band_names):
            method = self.norm_methods.get(name.lower(), "minmax")
            band_data = pre_tensor[:, c]                     # (t_event, H, W)
            finite_vals = band_data[np.isfinite(band_data)]  # flat array
 
            if len(finite_vals) == 0:
                logger.warning(f"  Band {name}: no finite values in pre-disturbance window")
                self._stats[name] = {"method": method, "min": 0.0, "max": 1.0,
                                     "mean": 0.0, "std": 1.0}
                continue
 
            stats = {
                "method": method,
                "min": float(np.min(finite_vals)),
                "max": float(np.max(finite_vals)),
                "mean": float(np.mean(finite_vals)),
                "std": float(np.std(finite_vals)),
            }
            self._stats[name] = stats
            logger.info(
                f"  {name:>10} [{method:>6}]  "
                f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
                f"mean={stats['mean']:.4f}  std={stats['std']:.4f}"
            )
 
        self._fitted = True
 
    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
 
    def transform(self, tensor: np.ndarray) -> np.ndarray:
        """
        Normalize a (T, C, H, W) or (C, H, W) tensor.
        NaN values are preserved (not touched).
        """
        self._check_fitted()
        out = tensor.copy().astype(np.float32)
        band_axis = 0 if tensor.ndim == 3 else 1
 
        for c, name in enumerate(self.band_names):
            if c >= tensor.shape[band_axis]:
                break
            stats = self._stats[name]
            method = stats["method"]
 
            if tensor.ndim == 3:
                band = out[c]
            else:
                band = out[:, c]
 
            mask = np.isfinite(band)
 
            if method == "minmax":
                vmin, vmax = stats["min"], stats["max"]
                rng = vmax - vmin
                if rng > 0:
                    band[mask] = (band[mask] - vmin) / rng
                else:
                    band[mask] = 0.0
                band[mask] = np.clip(band[mask], 0.0, 1.0)
 
            elif method == "shift":
                # Input range [−1, 1] → output [0, 1]
                band[mask] = (band[mask] + 1.0) / 2.0
                band[mask] = np.clip(band[mask], 0.0, 1.0)
 
            elif method == "zscore":
                mean, std = stats["mean"], stats["std"]
                if std > 0:
                    band[mask] = (band[mask] - mean) / std
                else:
                    band[mask] = 0.0
 
            if tensor.ndim == 3:
                out[c] = band
            else:
                out[:, c] = band
 
        return out
 
    def inverse_transform(self, tensor: np.ndarray) -> np.ndarray:
        """
        Reverse normalization back to original physical units.
        Used for counterfactual maps and metric computation.
        """
        self._check_fitted()
        out = tensor.copy().astype(np.float32)
        band_axis = 0 if tensor.ndim == 3 else 1
 
        for c, name in enumerate(self.band_names):
            if c >= tensor.shape[band_axis]:
                break
            stats = self._stats[name]
            method = stats["method"]
 
            if tensor.ndim == 3:
                band = out[c]
            else:
                band = out[:, c]
 
            mask = np.isfinite(band)
 
            if method == "minmax":
                vmin, vmax = stats["min"], stats["max"]
                band[mask] = band[mask] * (vmax - vmin) + vmin
 
            elif method == "shift":
                band[mask] = band[mask] * 2.0 - 1.0
 
            elif method == "zscore":
                mean, std = stats["mean"], stats["std"]
                band[mask] = band[mask] * std + mean
 
            if tensor.ndim == 3:
                out[c] = band
            else:
                out[:, c] = band
 
        return out
 
    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
 
    def save(self, path: str) -> None:
        """Save normalizer stats to JSON."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._stats, f, indent=2)
        logger.info(f"Normalizer stats saved → {path}")
 
    def load(self, path: str) -> None:
        """Load normalizer stats from JSON."""
        with open(path) as f:
            self._stats = json.load(f)
        self._fitted = True
        logger.info(f"Normalizer stats loaded ← {path}")
 
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Normalizer not fitted. Call normalizer.fit(tensor, t_event) first,\n"
                "or load from disk with normalizer.load(path)."
            )
 
 
def build_normalizers(
    config: Dict[str, Any],
    tensors: Dict[str, np.ndarray],
    processed_dir: str,
    force_refit: bool = False,
) -> Dict[str, EcoNormalizer]:
    """
    Build one normalizer per ecosystem, fitting on pre-disturbance data.
    Saves stats to processed/<ecosystem>/normalizer_stats.json.
 
    Returns dict: {"everglades": EcoNormalizer, "mississippi": EcoNormalizer}
    """
    normalizers = {}
 
    for ecosystem, tensor in tensors.items():
        stats_path = Path(processed_dir) / ecosystem / "normalizer_stats.json"
        norm = EcoNormalizer(config)
 
        if stats_path.exists() and not force_refit:
            logger.info(f"[{ecosystem}] Loading existing normalizer from {stats_path}")
            norm.load(str(stats_path))
        else:
            t_event = config["ecosystems"][ecosystem]["t_event"]
            logger.info(f"[{ecosystem}] Fitting normalizer (t_event={t_event}) ...")
            norm.fit(tensor, t_event)
            norm.save(str(stats_path))

        normalizers[ecosystem] = norm
 
    return normalizers
 