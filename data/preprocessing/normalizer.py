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
 
SeasonalClimatology:
  Computes per-pixel, per-season (quarter-of-year) mean and std from
  pre-event years. Used as a soft lower-bound during CF rollout to prevent
  compounding winter dips across autoregressive iterations.
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
 
 
class SeasonalClimatology:
    """
    Per-pixel, per-season (quarter-of-year) climatology computed from
    pre-disturbance years.
 
    Purpose: provide a soft lower-bound floor for CF predictions so that
    autoregressive rollout does not compound seasonal dips quarter after
    quarter.  When the model predicts a winter low that is n_sigma below
    the historical seasonal mean for that pixel, clamp it back up.
 
    Usage:
        clim = SeasonalClimatology(config)
        clim.fit(norm_tensor, t_event=6, start_quarter=1)   # quarter 1=Q1
 
        # In rollout, after generating a predicted frame at step `step`:
        quarter = (start_quarter + t_event + step - 1) % 4   # 0-indexed
        floor   = clim.get_floor(quarter, n_sigma=2.0)        # (C, H, W)
        pred_frame = np.maximum(pred_frame, floor)
    """
 
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_bands = config["bands"]["count"]
        # seasonal_mean[q] = (C, H, W) mean for quarter q (0=Q1, 1=Q2, 2=Q3, 3=Q4)
        self.seasonal_mean: Dict[int, np.ndarray] = {}
        self.seasonal_std:  Dict[int, np.ndarray] = {}
        self._fitted = False
 
    def fit(
        self,
        norm_tensor: np.ndarray,
        t_event: int,
        start_quarter: int = 1,
    ) -> None:
        """
        Compute seasonal statistics from normalized pre-disturbance data.
 
        Args:
            norm_tensor   : (T, C, H, W) NORMALIZED full time-series
            t_event       : only use frames [0, t_event)
            start_quarter : calendar quarter of frame 0 (1=Q1, 2=Q2, ...)
        """
        pre = norm_tensor[:t_event]   # (t_event, C, H, W)
        T = pre.shape[0]
 
        # Group frames by quarter-of-year (0-indexed: 0=Q1, 1=Q2, 2=Q3, 3=Q4)
        by_quarter: Dict[int, list] = {0: [], 1: [], 2: [], 3: []}
        for t in range(T):
            q = ((start_quarter - 1) + t) % 4
            by_quarter[q].append(pre[t])  # (C, H, W)
 
        for q in range(4):
            frames = by_quarter[q]
            if len(frames) == 0:
                # No data for this quarter — use global mean
                self.seasonal_mean[q] = np.nanmean(pre, axis=0).astype(np.float32)
                self.seasonal_std[q]  = np.nanstd(pre,  axis=0).astype(np.float32)
            else:
                stack = np.stack(frames, axis=0)   # (n, C, H, W)
                self.seasonal_mean[q] = np.nanmean(stack, axis=0).astype(np.float32)
                self.seasonal_std[q]  = np.nanstd(stack,  axis=0).astype(np.float32)
            logger.info(
                f"  SeasonalClimatology Q{q+1}: {len(frames)} frames | "
                f"NDVI mean={self.seasonal_mean[q][self.config['bands']['indices']['ndvi']].mean():.3f}"
            )
 
        self._fitted = True
 
    def get_floor(self, quarter: int, n_sigma: float = 2.0) -> np.ndarray:
        """
        Return per-pixel floor for a given quarter.
 
        floor = seasonal_mean - n_sigma * seasonal_std
 
        Any prediction below this floor is clamped up to the floor during
        autoregressive rollout to prevent seasonal-dip compounding.
 
        Args:
            quarter : 0-indexed quarter (0=Q1, 1=Q2, 2=Q3, 3=Q4)
            n_sigma : how many std below mean before clamping (default 2.0)
 
        Returns:
            (C, H, W) float32 floor array in normalised space
        """
        if not self._fitted:
            raise RuntimeError("SeasonalClimatology not fitted. Call fit() first.")
        q = int(quarter) % 4
        floor = self.seasonal_mean[q] - n_sigma * self.seasonal_std[q]
        return floor.astype(np.float32)
 
    def save(self, path: str) -> None:
        """Save climatology arrays as .npz."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {}
        for q in range(4):
            save_dict[f"mean_q{q}"] = self.seasonal_mean[q]
            save_dict[f"std_q{q}"]  = self.seasonal_std[q]
        np.savez(path, **save_dict)
        logger.info(f"SeasonalClimatology saved → {path}")
 
    def load(self, path: str) -> None:
        """Load climatology from .npz."""
        data = np.load(path)
        for q in range(4):
            self.seasonal_mean[q] = data[f"mean_q{q}"]
            self.seasonal_std[q]  = data[f"std_q{q}"]
        self._fitted = True
        logger.info(f"SeasonalClimatology loaded ← {path}")
 
 
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