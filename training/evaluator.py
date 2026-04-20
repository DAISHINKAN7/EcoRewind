"""
evaluator.py  (v2 — uses optical validity mask for correct R² computation)
---------------------------------------------------------------------------

FIX D: SAR fill values (0.0) are finite, so torch.isfinite(target) returns True
for SAR fill pixels. The evaluator was computing R² against those fill values,
producing Blue R²=-825, SAR R²=-102.

FIX: Use the validity channel (8th patch channel) to mask evaluation.
validity=1 means the optical data is real. validity=0 means it's a fill.
Only evaluate R² and MSE on validity=1 pixels.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelEvaluator:

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
        self.model.eval()

        sum_sq_err = np.zeros(self.n_bands, dtype=np.float64)
        sum_sq_tot = np.zeros(self.n_bands, dtype=np.float64)
        sum_target = np.zeros(self.n_bands, dtype=np.float64)
        n_pixels = np.zeros(self.n_bands, dtype=np.int64)

        for batch in loader:
            inp = batch["input"].to(self.device)
            target_full = batch["target"].to(self.device)

            # FIX D: Extract validity channel (optical-only validity)
            if target_full.shape[2] > self.n_bands:
                validity = target_full[:, :, self.n_bands:]   # (B, T, 1, H, W)
                target = target_full[:, :, :self.n_bands]
            else:
                validity = None
                target = target_full

            pred = self.model(inp)

            # FIX D: Use optical validity mask (not isfinite) for evaluation
            # validity=1 means real data, validity=0 means fill value
            if validity is not None:
                # Expand (B, T, 1, H, W) → (B, T, C, H, W)
                valid_mask = validity.expand_as(target).bool()
            else:
                # Fallback: use isfinite (old behavior)
                valid_mask = torch.isfinite(target)

            # Additional safety: exclude SAR from R² computation
            # SAR has only 12.5% valid pixels — R² is unreliable with this coverage
            # We'll still report it but note it separately
            for c in range(self.n_bands):
                if c >= pred.shape[2]:
                    continue

                p_c = pred[:, :, c].cpu().numpy().ravel()
                t_c = target[:, :, c].cpu().numpy().ravel()
                v_c = valid_mask[:, :, c].cpu().numpy().ravel().astype(bool)

                p_c = p_c[v_c]
                t_c = t_c[v_c]

                if len(t_c) == 0:
                    continue

                sum_sq_err[c] += np.sum((p_c - t_c) ** 2)
                sum_target[c] += np.sum(t_c)
                n_pixels[c] += len(t_c)

        # Second pass for R²
        n_safe = np.maximum(n_pixels, 1)
        mean_target = sum_target / n_safe

        for batch in loader:
            inp = batch["input"].to(self.device)
            target_full = batch["target"].to(self.device)
            if target_full.shape[2] > self.n_bands:
                validity = target_full[:, :, self.n_bands:]
                target = target_full[:, :, :self.n_bands]
            else:
                validity = None
                target = target_full

            if validity is not None:
                valid_mask = validity.expand_as(target).bool()
            else:
                valid_mask = torch.isfinite(target)

            for c in range(self.n_bands):
                t_c = target[:, :, c].cpu().numpy().ravel()
                v_c = valid_mask[:, :, c].cpu().numpy().ravel().astype(bool)
                t_c = t_c[v_c]
                if len(t_c) == 0:
                    continue
                sum_sq_tot[c] += np.sum((t_c - mean_target[c]) ** 2)

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

        total_err = np.sum(sum_sq_err)
        total_n = np.sum(n_pixels)
        results["rmse_overall"] = float(np.sqrt(total_err / max(total_n, 1)))
        results["ndvi_mse"] = results.get("mse_ndvi", float("nan"))
        results["sar_mse"] = results.get("mse_sar_vv", float("nan"))

        self._print_results(results, n_pixels)
        return results

    def _print_results(self, results: Dict[str, float], n_pixels: np.ndarray):
        logger.info("\n" + "="*50)
        logger.info("  Model Evaluation Results")
        logger.info("="*50)
        logger.info(f"  Overall RMSE: {results.get('rmse_overall', float('nan')):.6f}")
        logger.info(f"\n        Band         MSE        R²    N_valid_px")
        logger.info("  " + "-"*50)
        for i, name in enumerate(self.band_names):
            if i >= self.n_bands:
                break
            mse = results.get(f"mse_{name.lower()}", float("nan"))
            r2 = results.get(f"r2_{name.lower()}", float("nan"))
            n = n_pixels[i] if i < len(n_pixels) else 0
            note = " ← sparse (SAR)" if name == "SAR_VV" and r2 < -1.0 else ""
            logger.info(f"  {name:>10}    {mse:>8.6f}   {r2:>8.4f}    {n:>10,}{note}")
        logger.info("="*50 + "\n")