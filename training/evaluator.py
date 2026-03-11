"""
evaluator.py
------------
Evaluation metrics for the pre-disturbance reconstruction quality.
 
Metrics computed on held-out pre-disturbance patches:
  - Per-band MSE
  - NDVI reconstruction MSE (most ecologically meaningful)
  - Overall RMSE
  - R² per band
 
These are model quality metrics — not ecological impact metrics.
Ecological metrics (hectares impacted, carbon loss) are in inference/metrics.py.
"""
 
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
 
logger = logging.getLogger(__name__)
 
 
class ModelEvaluator:
    """Evaluate reconstruction quality on a DataLoader."""
 
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device
        self.band_names = config["bands"]["names"]
        self.n_bands = config["bands"]["count"]
        self.ndvi_idx = config["bands"]["indices"]["ndvi"]
        self.sar_idx = config["bands"]["indices"]["sar_vv"]
 
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run evaluation on all batches in the loader.
 
        Returns dict with:
          - "mse_<band_name>"  per-band MSE
          - "rmse_overall"     overall RMSE
          - "r2_<band_name>"   R² per band
          - "ndvi_mse"         shortcut
          - "sar_mse"          shortcut
        """
        self.model.eval()
 
        # Accumulators per band
        sum_sq_err = np.zeros(self.n_bands, dtype=np.float64)
        sum_sq_tot = np.zeros(self.n_bands, dtype=np.float64)
        sum_target = np.zeros(self.n_bands, dtype=np.float64)
        n_pixels = np.zeros(self.n_bands, dtype=np.int64)
 
        all_preds = []
        all_targets = []
 
        for batch in loader:
            inp = batch["input"].to(self.device)
            target = batch["target"].to(self.device)   # (B, T_out, C, H, W)
 
            # Do NOT strip validity channel from inp — the model was trained with it
            # and expects the same channel count at inference time.
            # The metric loop below already only iterates over range(n_bands) so any
            # extra channels in target (validity mask) are naturally ignored.
 
            pred = self.model(inp)   # (B, T_out, C, H, W)
 
            # Only evaluate on valid (non-NaN) target pixels
            valid_mask = torch.isfinite(target)
 
            for c in range(self.n_bands):
                if c >= pred.shape[2]:
                    continue
                p_c = pred[:, :, c].cpu().numpy().ravel()
                t_c = target[:, :, c].cpu().numpy().ravel()
                v_c = valid_mask[:, :, c].cpu().numpy().ravel()
 
                p_c = p_c[v_c]
                t_c = t_c[v_c]
 
                if len(t_c) == 0:
                    continue
 
                sum_sq_err[c] += np.sum((p_c - t_c) ** 2)
                sum_target[c] += np.sum(t_c)
                n_pixels[c] += len(t_c)
 
        # Compute per-band means for R²
        n_safe = np.maximum(n_pixels, 1)
        mean_target = sum_target / n_safe
 
        # Need second pass for R² (sum of squares total needs mean)
        # Quick approximation: use accumulated values
        # Re-run briefly for R²
        for batch in loader:
            inp = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            valid_mask = torch.isfinite(target)
            for c in range(self.n_bands):
                t_c = target[:, :, c].cpu().numpy().ravel()
                v_c = valid_mask[:, :, c].cpu().numpy().ravel()
                t_c = t_c[v_c]
                if len(t_c) == 0:
                    continue
                sum_sq_tot[c] += np.sum((t_c - mean_target[c]) ** 2)
 
        # Build results dict
        results = {}
        for c, name in enumerate(self.band_names):
            if c >= self.n_bands:
                break
            n = n_pixels[c]
            if n == 0:
                continue
            mse = float(sum_sq_err[c] / n)
            r2 = 1.0 - (sum_sq_err[c] / max(sum_sq_tot[c], 1e-10))
            results[f"mse_{name.lower()}"] = mse
            results[f"r2_{name.lower()}"] = float(r2)
 
        # Overall RMSE across all bands
        total_err = np.sum(sum_sq_err)
        total_n = np.sum(n_pixels)
        results["rmse_overall"] = float(np.sqrt(total_err / max(total_n, 1)))
        results["ndvi_mse"] = results.get(f"mse_ndvi", float("nan"))
        results["sar_mse"] = results.get(f"mse_sar_vv", float("nan"))
 
        self._print_results(results)
        return results
 
    def _print_results(self, results: Dict[str, float]):
        logger.info("\n" + "="*50)
        logger.info("  Model Evaluation Results")
        logger.info("="*50)
        logger.info(f"  Overall RMSE: {results.get('rmse_overall', float('nan')):.6f}")
        logger.info(f"\n  {'Band':>10}  {'MSE':>10}  {'R²':>8}")
        logger.info("  " + "-" * 32)
        for name in self.band_names:
            mse = results.get(f"mse_{name.lower()}", float("nan"))
            r2 = results.get(f"r2_{name.lower()}", float("nan"))
            logger.info(f"  {name:>10}  {mse:>10.6f}  {r2:>8.4f}")
        logger.info("="*50 + "\n")
 