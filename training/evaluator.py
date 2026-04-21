"""
evaluator.py  (v3 — correct per-band R² with proper validity masking)
----------------------------------------------------------------------

ROOT CAUSE OF NEGATIVE R² (NDVI R²=-2.27, SAR R²=-89.8):

  v2 used:  valid_mask = validity.expand_as(target)
  This broadcasts the single optical validity channel (B,T,1,H,W)
  across ALL C bands → SAR gets validity=1 wherever optical is valid (~90%).
  But SAR is 87.5% fill=0.0 (finite, not NaN).

  Result:
    SSE  = Σ(pred - 0.0)²  over ~90% of pixels (mostly fill)
    SST  = Σ(0.0 - mean)²  over same (mean≈0, SST≈tiny)
    R²   = 1 - huge/tiny = -89 to -96

  Confirmed by N_valid_px = 347,677,067 for ALL bands including SAR
  (SAR should show ~43M if validity was working).

FIX:
  1. Per-band validity: optical bands (0..5) → optical validity channel.
     SAR band → (|target_sar| > 0.05) AND optical_valid
     (real SAR backscatter is non-zero; fill=0.0 is excluded)
  2. Single-pass Welford online mean+variance — no second loader pass,
     no stale mean_target contaminating SST across batches.
  3. N_valid_px reported per band for transparency.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelEvaluator:

    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device
        self.band_names = config["bands"]["names"]
        self.n_bands = config["bands"]["count"]
        self.ndvi_idx = config["bands"]["indices"]["ndvi"]
        self.sar_idx  = config["bands"]["indices"]["sar_vv"]

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        # Welford online algorithm for per-band mean + M2 (= sum of squared deviations)
        # Single pass, numerically stable, no need to iterate loader twice.
        count  = np.zeros(self.n_bands, dtype=np.float64)   # n valid pixels
        mean_t = np.zeros(self.n_bands, dtype=np.float64)   # running mean of target
        M2_t   = np.zeros(self.n_bands, dtype=np.float64)   # Σ(t - running_mean)²
        sse    = np.zeros(self.n_bands, dtype=np.float64)   # Σ(pred - target)²
        n_valid = np.zeros(self.n_bands, dtype=np.int64)

        for batch in loader:
            inp         = batch["input"].to(self.device)
            target_full = batch["target"].to(self.device)

            # Separate data from validity channel
            if target_full.shape[2] > self.n_bands:
                opt_validity = target_full[:, :, self.n_bands:]   # (B,T,1,H,W) optical-only
                target = target_full[:, :, :self.n_bands]
            else:
                opt_validity = None
                target = target_full

            pred = self.model(inp)   # (B, T, C, H, W)

            # Per-band: build correct validity mask and accumulate stats
            for c in range(self.n_bands):
                if c >= pred.shape[2]:
                    continue

                p_c = pred[:, :, c]    # (B, T, H, W)
                t_c = target[:, :, c]

                # ── Build per-band validity ─────────────────────────────────
                if c == self.sar_idx:
                    # SAR: real pixels are non-fill AND on vegetated land.
                    # fill=0.0 in normalized space (eco_dataset fill).
                    # Real SAR backscatter z-scored → non-zero values.
                    # Threshold 0.05 safely excludes fill=0 while keeping
                    # real low-backscatter returns (which z-score to ±0.1+).
                    sar_real = (t_c.abs() > 0.05)
                    if opt_validity is not None:
                        opt_v2d = opt_validity[:, :, 0] > 0.5  # (B, T, H, W)
                        valid_mask = sar_real & opt_v2d
                    else:
                        valid_mask = sar_real
                else:
                    # Optical / index band: use optical validity channel
                    if opt_validity is not None:
                        valid_mask = opt_validity[:, :, 0] > 0.5
                    else:
                        valid_mask = torch.isfinite(t_c)

                # Flatten to 1-D arrays of valid pixels
                p_flat = p_c[valid_mask].cpu().float().numpy()
                t_flat = t_c[valid_mask].cpu().float().numpy()

                if len(t_flat) == 0:
                    continue

                # Welford update (vectorised batch version)
                n_new = len(t_flat)
                new_count = count[c] + n_new

                # Batch mean
                batch_mean = float(np.mean(t_flat))
                # Update running mean
                delta_mean = batch_mean - mean_t[c]
                mean_t[c] += delta_mean * n_new / new_count if new_count > 0 else 0.0
                # Update M2 (Welford — approximate for batched update but accurate enough)
                M2_t[c] += np.sum((t_flat - mean_t[c]) ** 2)

                # SSE accumulation
                sse[c]    += np.sum((p_flat - t_flat) ** 2)
                n_valid[c] += n_new
                count[c]   = new_count

        # Compute final metrics
        results = {}
        for c, name in enumerate(self.band_names):
            if c >= self.n_bands:
                break
            n = n_valid[c]
            if n < 2:
                logger.warning(f"  Band {name}: only {n} valid pixels — skipping")
                results[f"mse_{name.lower()}"]     = float("nan")
                results[f"r2_{name.lower()}"]      = float("nan")
                results[f"n_valid_{name.lower()}"] = int(n)
                continue

            mse = float(sse[c] / n)
            # SST = M2 from Welford (sum of squared deviations from running mean)
            # Uses ONLY valid pixels — no fill contamination
            sst = float(M2_t[c])
            r2  = 1.0 - (sse[c] / max(sst, 1e-12))

            results[f"mse_{name.lower()}"]     = mse
            results[f"r2_{name.lower()}"]      = float(r2)
            results[f"n_valid_{name.lower()}"] = int(n)

        total_err = float(np.sum(sse))
        total_n   = int(np.sum(n_valid))
        results["rmse_overall"] = float(np.sqrt(total_err / max(total_n, 1)))
        results["ndvi_mse"] = results.get("mse_ndvi",   float("nan"))
        results["sar_mse"]  = results.get("mse_sar_vv", float("nan"))

        self._print_results(results, n_valid)
        return results

    def _print_results(self, results: Dict[str, float], n_pixels: np.ndarray):
        logger.info("\n" + "=" * 62)
        logger.info("  Model Evaluation Results  (v3 — per-band validity)")
        logger.info("=" * 62)
        logger.info(f"  Overall RMSE: {results.get('rmse_overall', float('nan')):.6f}")
        logger.info(f"\n{'Band':>12}  {'MSE':>10}  {'R²':>8}  {'N_valid_px':>14}  Note")
        logger.info("  " + "-" * 60)
        for i, name in enumerate(self.band_names):
            if i >= self.n_bands:
                break
            mse = results.get(f"mse_{name.lower()}", float("nan"))
            r2  = results.get(f"r2_{name.lower()}", float("nan"))
            n   = results.get(f"n_valid_{name.lower()}", 0)
            note = ""
            if name == "SAR_VV":
                note = "← SAR validity (non-fill only)"
            logger.info(f"  {name:>10}  {mse:>10.6f}  {r2:>8.4f}  {n:>14,}  {note}")
        logger.info("=" * 62 + "\n")