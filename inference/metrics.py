"""
metrics.py
----------
Quantitative ecological impact metrics from counterfactual trajectories.
 
Computes:
  1. ΔNDVI damage maps (counterfactual − actual)
  2. Impacted area in hectares (ΔNDVI > threshold)
  3. Carbon stock loss estimate (tCO₂-equivalent)
  4. Recovery rate (slope of NDVI toward counterfactual)
  5. Time-to-convergence (when actual rejoins CF 95% CI)
"""
 
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
 
logger = logging.getLogger(__name__)
 
 
class EcologicalMetrics:
    """Compute and report all ecological impact metrics."""
 
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ndvi_idx = config["bands"]["indices"]["ndvi"]
        self.ndwi_idx = config["bands"]["indices"]["ndwi"]
        self.sar_idx = config["bands"]["indices"]["sar_vv"]
 
        self.impact_threshold = config["inference"]["impact_threshold"]
        self.carbon_factor = config["inference"]["carbon_factor"]
        self.convergence_ci = config["inference"]["convergence_ci"]
 
    def compute_all(
        self,
        ecosystem: str,
        cf_results: Dict[str, np.ndarray],
        pixel_size_m: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Compute all metrics from counterfactual generation output.
 
        Args:
            ecosystem   : "everglades" or "mississippi"
            cf_results  : output of CounterfactualGenerator.generate()
                          keys: "counterfactual", "actual_post", "delta"
            pixel_size_m: pixel size in meters (default 10m)
 
        Returns:
            dict with all metrics + time-series arrays
        """
        cf = cf_results["counterfactual"]      # (T_post, C, H, W) physical units
        actual = cf_results["actual_post"]     # (T_post, C, H, W)
        delta = cf_results["delta"]            # (T_post, C, H, W) CF − actual
 
        T_post, C, H, W = cf.shape
        pixel_area_ha = (pixel_size_m ** 2) / 10_000.0   # m² → hectares
 
        logger.info(f"\n[{ecosystem.upper()}] Computing ecological metrics...")
 
        results = {
            "ecosystem": ecosystem,
            "n_post_quarters": T_post,
            "pixel_size_m": pixel_size_m,
            "pixel_area_ha": pixel_area_ha,
        }
 
        # 1. ΔNDVI map at each timestep
        delta_ndvi = delta[:, self.ndvi_idx]     # (T_post, H, W)
        delta_ndwi = delta[:, self.ndwi_idx]
        delta_sar = delta[:, self.sar_idx]
 
        results["delta_ndvi_mean_timeseries"] = self._nanmean_timeseries(delta_ndvi)
        results["delta_ndwi_mean_timeseries"] = self._nanmean_timeseries(delta_ndwi)
        results["delta_sar_mean_timeseries"] = self._nanmean_timeseries(delta_sar)
 
        # 2. Impacted area
        impacted_ha_ts = self._impacted_area_timeseries(
            delta_ndvi, self.impact_threshold, pixel_area_ha
        )
        results["impacted_ha_timeseries"] = impacted_ha_ts
        results["impacted_ha_max"] = float(np.max(impacted_ha_ts)) if impacted_ha_ts else 0.0
        results["impacted_ha_mean"] = float(np.mean(impacted_ha_ts)) if impacted_ha_ts else 0.0
        logger.info(
            f"  Impacted area (ΔNDVI > {self.impact_threshold}):\n"
            f"    Max:  {results['impacted_ha_max']:.1f} ha\n"
            f"    Mean: {results['impacted_ha_mean']:.1f} ha"
        )
 
        # 3. Carbon loss estimate
        carbon_loss_tc = self._carbon_loss(
            delta_ndvi[0] if T_post > 0 else np.zeros((H, W)),
            pixel_area_ha,
        )
        results["carbon_loss_tco2_eq"] = float(carbon_loss_tc)
        logger.info(f"  Estimated carbon loss: {carbon_loss_tc:.1f} tCO₂-eq")
 
        # 4. Recovery rate
        cf_ndvi = cf[:, self.ndvi_idx]        # (T_post, H, W)
        actual_ndvi = actual[:, self.ndvi_idx]
        recovery = self._recovery_rate(actual_ndvi, cf_ndvi)
        results.update(recovery)
        logger.info(
            f"  Recovery rate:\n"
            f"    Actual NDVI slope:       {recovery.get('actual_ndvi_slope', float('nan')):.4f}/quarter\n"
            f"    Counterfactual NDVI slope: {recovery.get('cf_ndvi_slope', float('nan')):.4f}/quarter\n"
            f"    Gap closing rate:         {recovery.get('gap_closing_rate', float('nan')):.4f}/quarter"
        )
 
        # 5. Time-to-convergence
        ttc = self._time_to_convergence(actual_ndvi, cf_ndvi)
        results["time_to_convergence_quarters"] = ttc
        if ttc is not None:
            logger.info(f"  Time-to-convergence: {ttc} quarters (~{ttc/4:.1f} years)")
        else:
            logger.info("  Time-to-convergence: not reached within observation window")
 
        # 6. Summary stats
        results["summary"] = self._build_summary(results)
        logger.info(f"\n  === SUMMARY ===\n{self._format_summary(results['summary'])}")
 
        return results
 
    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
 
    def _nanmean_timeseries(self, arr: np.ndarray) -> List[float]:
        """Compute mean per timestep, ignoring NaN."""
        result = []
        for t in range(arr.shape[0]):
            vals = arr[t][np.isfinite(arr[t])]
            result.append(float(np.mean(vals)) if len(vals) > 0 else float("nan"))
        return result
 
    def _impacted_area_timeseries(
        self,
        delta_ndvi: np.ndarray,
        threshold: float,
        pixel_area_ha: float,
    ) -> List[float]:
        """
        At each timestep, count pixels where ΔNDVI > threshold.
        Note: positive ΔNDVI = CF > actual = vegetation loss in actual.
        """
        result = []
        for t in range(delta_ndvi.shape[0]):
            frame = delta_ndvi[t]
            n_impacted = np.sum(np.isfinite(frame) & (frame > threshold))
            result.append(float(n_impacted * pixel_area_ha))
        return result
 
    def _carbon_loss(
        self,
        delta_ndvi_t0: np.ndarray,   # (H, W) first post-event quarter
        pixel_area_ha: float,
    ) -> float:
        """
        Rough carbon loss estimate using allometric relationship.
 
        ΔNDVI × carbon_factor (tCO₂/ha per NDVI unit) × pixel area × n_pixels
 
        carbon_factor from config: default 85 tCO₂-eq/ha (mangrove/wetland estimate)
        Source: Murdiyarso et al. 2015 (mangrove carbon stocks)
        """
        valid = delta_ndvi_t0[np.isfinite(delta_ndvi_t0) & (delta_ndvi_t0 > 0)]
        if len(valid) == 0:
            return 0.0
 
        # Total carbon loss = sum over impacted pixels of ΔNDVI × carbon_factor × area
        # Normalized: ΔNDVI ∈ [0,1] represents fraction of maximum biomass lost
        total_loss = float(np.sum(valid) * self.carbon_factor * pixel_area_ha)
        return total_loss
 
    def _recovery_rate(
        self,
        actual_ndvi: np.ndarray,   # (T_post, H, W)
        cf_ndvi: np.ndarray,
    ) -> Dict[str, float]:
        """
        Estimate recovery rate as:
          - actual_ndvi_slope: linear fit slope of actual NDVI mean over time
          - cf_ndvi_slope: linear fit slope of CF NDVI mean over time
          - gap_closing_rate: rate at which actual approaches counterfactual
        """
        T = actual_ndvi.shape[0]
        if T < 2:
            return {"actual_ndvi_slope": float("nan"),
                    "cf_ndvi_slope": float("nan"),
                    "gap_closing_rate": float("nan")}
 
        t_axis = np.arange(T, dtype=np.float64)
 
        # Mean NDVI per timestep
        actual_mean = np.array([
            np.nanmean(actual_ndvi[t]) for t in range(T)
        ])
        cf_mean = np.array([
            np.nanmean(cf_ndvi[t]) for t in range(T)
        ])
        gap = cf_mean - actual_mean   # positive = recovery needed
 
        # Linear fits
        def safe_slope(y: np.ndarray) -> float:
            valid = np.isfinite(y)
            if valid.sum() < 2:
                return float("nan")
            coeffs = np.polyfit(t_axis[valid], y[valid], 1)
            return float(coeffs[0])
 
        return {
            "actual_ndvi_slope": safe_slope(actual_mean),
            "cf_ndvi_slope": safe_slope(cf_mean),
            "gap_closing_rate": -safe_slope(gap),   # negative gap slope = gap closing
            "actual_ndvi_timeseries": actual_mean.tolist(),
            "cf_ndvi_timeseries": cf_mean.tolist(),
            "gap_timeseries": gap.tolist(),
        }
 
    def _time_to_convergence(
        self,
        actual_ndvi: np.ndarray,   # (T_post, H, W)
        cf_ndvi: np.ndarray,
        ci: Optional[float] = None,
    ) -> Optional[int]:
        """
        Time-to-convergence: first quarter where actual NDVI mean
        falls within the CI of the counterfactual NDVI distribution.
        """
        if ci is None:
            ci = self.convergence_ci
 
        T = actual_ndvi.shape[0]
        z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(ci, 1.960)
 
        for t in range(T):
            cf_t = cf_ndvi[t][np.isfinite(cf_ndvi[t])]
            act_t = actual_ndvi[t][np.isfinite(actual_ndvi[t])]
 
            if len(cf_t) < 2 or len(act_t) < 2:
                continue
 
            cf_mean = np.mean(cf_t)
            cf_std = np.std(cf_t)
            act_mean = np.mean(act_t)
 
            # Check if actual mean is within CI of CF distribution
            lower = cf_mean - z * cf_std
            upper = cf_mean + z * cf_std
            if lower <= act_mean <= upper:
                return t
 
        return None   # not converged within observation window
 
    def _build_summary(self, results: Dict) -> Dict:
        return {
            "ecosystem": results["ecosystem"],
            "n_post_quarters": results["n_post_quarters"],
            "impacted_ha_max": results.get("impacted_ha_max", 0.0),
            "impacted_ha_mean": results.get("impacted_ha_mean", 0.0),
            "carbon_loss_tco2_eq": results.get("carbon_loss_tco2_eq", 0.0),
            "actual_ndvi_slope_per_quarter": results.get("actual_ndvi_slope", float("nan")),
            "gap_closing_rate_per_quarter": results.get("gap_closing_rate", float("nan")),
            "time_to_convergence_quarters": results.get("time_to_convergence_quarters"),
        }
 
    def _format_summary(self, summary: Dict) -> str:
        lines = []
        for k, v in summary.items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.4f}")
            else:
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)
 
    def save(self, results: Dict, output_path: str):
        """Save metrics to JSON (strip numpy arrays for JSON serialization)."""
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            raise TypeError(f"Not serializable: {type(obj)}")
 
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=serialize)
        logger.info(f"Metrics saved → {output_path}")