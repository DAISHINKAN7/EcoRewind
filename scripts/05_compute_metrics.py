"""
05_compute_metrics.py
---------------------
Step 5: Load saved counterfactual arrays and compute all ecological
        metrics, then generate a consolidated report.

BUG FIX: Now reads from model-specific subdirectory:
  processed/<eco>/<model_arch>/counterfactual.npy   (was: processed/<eco>/counterfactual.npy)

Run after 04_generate_counterfactual.py.
Can be run standalone if .npy arrays already exist.

Outputs:
  - outputs/metrics/consolidated_report_<model>.json
  - outputs/metrics/comparison_table_<model>.csv

Usage:
    python scripts/05_compute_metrics.py --model unet_patchtst
    python scripts/05_compute_metrics.py --model unet_mamba
    python scripts/05_compute_metrics.py --model unet_patchtst --ecosystem everglades
    python scripts/05_compute_metrics.py --all-models   # iterate all available
"""

import sys
import os
import argparse
import logging
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_MODELS = ["convlstm", "unet_temporal", "eco_transformer", "unet_mamba", "unet_patchtst"]


def run_for_model(config, model_arch: str, ecosystems: list,
                  processed_dir: Path, metrics_dir: Path, maps_dir: Path):
    """Load and process counterfactual arrays for one model."""

    from inference.metrics import EcologicalMetrics
    from visualization.maps import plot_trajectories, plot_recovery_metrics

    metrics_engine = EcologicalMetrics(config)
    all_results = {}

    for eco in ecosystems:
        logger.info(f"\n--- {eco.upper()} / {model_arch} ---")

        # Model-specific array paths
        eco_arrays_dir = processed_dir / eco / model_arch
        delta_path  = eco_arrays_dir / "delta.npy"
        cf_path     = eco_arrays_dir / "counterfactual.npy"
        actual_path = eco_arrays_dir / "actual_post.npy"

        if not delta_path.exists():
            # Fall back to legacy (non-model-specific) path for backward compat
            legacy_delta = processed_dir / eco / "delta.npy"
            if legacy_delta.exists():
                logger.warning(
                    f"  No model-specific arrays at {eco_arrays_dir}/\n"
                    f"  Falling back to legacy path {legacy_delta}\n"
                    f"  Re-run 04_generate_counterfactual.py --model {model_arch} to fix."
                )
                eco_arrays_dir = processed_dir / eco
                delta_path  = eco_arrays_dir / "delta.npy"
                cf_path     = eco_arrays_dir / "counterfactual.npy"
                actual_path = eco_arrays_dir / "actual_post.npy"
            else:
                logger.warning(
                    f"  No arrays found for {eco}/{model_arch}.\n"
                    f"  Run: python scripts/04_generate_counterfactual.py"
                    f" --model {model_arch} --ecosystem {eco}"
                )
                continue

        delta  = np.load(str(delta_path))
        cf     = np.load(str(cf_path))
        actual = np.load(str(actual_path))
        logger.info(f"  Loaded arrays from {eco_arrays_dir}/")
        logger.info(f"  delta={delta.shape}, cf={cf.shape}, actual={actual.shape}")

        cf_results = {
            "counterfactual": cf,
            "actual_post":    actual,
            "delta":          delta,
        }
        # Load uncertainty if present
        unc_path = eco_arrays_dir / "uncertainty.npy"
        if unc_path.exists():
            cf_results["uncertainty"] = np.load(str(unc_path))
            logger.info(f"  Loaded uncertainty maps")

        # Compute metrics
        results = metrics_engine.compute_all(eco, cf_results, pixel_size_m=10.0)
        all_results[eco] = results

        # Plots go into model-specific subdir
        eco_maps_dir = maps_dir / eco / model_arch
        eco_maps_dir.mkdir(parents=True, exist_ok=True)
        plot_trajectories(results, config, eco, str(maps_dir / eco))
        plot_recovery_metrics(results, config, eco, str(maps_dir / eco))

        # Per-ecosystem metrics JSON
        eco_metrics_path = metrics_dir / f"{eco}_{model_arch}_metrics_final.json"
        metrics_engine.save(results, str(eco_metrics_path))
        logger.info(f"  Metrics → {eco_metrics_path}")

    return all_results


def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info(" ECO-REWIND: Step 5 — Compute & Consolidate Metrics")
    logger.info("=" * 60)

    processed_dir = Path(config["data"]["processed_dir"])
    metrics_dir   = Path(config["outputs"]["metrics"])
    maps_dir      = Path(config["outputs"]["maps"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ecosystems = (
        [args.ecosystem] if args.ecosystem
        else list(config["ecosystems"].keys())
    )

    # Which models to process
    if args.all_models:
        models_to_run = []
        for m in ALL_MODELS:
            # Include model if arrays exist for at least one ecosystem
            for eco in ecosystems:
                if (processed_dir / eco / m / "delta.npy").exists():
                    if m not in models_to_run:
                        models_to_run.append(m)
                    break
        if not models_to_run:
            logger.warning("No model-specific arrays found. Run 04_generate_counterfactual.py first.")
            return
        logger.info(f"  Auto-detected models with arrays: {models_to_run}")
    elif args.model:
        models_to_run = [args.model]
    else:
        # Default: use architecture from config
        models_to_run = [config["model"]["architecture"]]

    # Run metrics for each model
    all_model_results = {}
    for model_arch in models_to_run:
        logger.info(f"\n{'='*50}")
        logger.info(f"  Processing model: {model_arch}")
        logger.info(f"{'='*50}")
        results = run_for_model(
            config, model_arch, ecosystems,
            processed_dir, metrics_dir, maps_dir
        )
        if results:
            all_model_results[model_arch] = results
            _save_consolidated_report(results, metrics_dir, config, model_arch)
            _save_comparison_csv(results, metrics_dir, model_arch)

    # Cross-model comparison table (if multiple models processed)
    if len(all_model_results) > 1:
        _save_cross_model_comparison(all_model_results, metrics_dir, ecosystems)

    logger.info("\n" + "=" * 60)
    logger.info(" Metrics computation complete!")
    logger.info(f"  Reports: {metrics_dir}/")
    logger.info(f"  Maps:    {maps_dir}/")
    logger.info("=" * 60)


def _save_consolidated_report(results: dict, metrics_dir: Path, config: dict, model_arch: str):
    report = {
        "project":    "ECO-REWIND",
        "model":      model_arch,
        "ecosystems": {},
    }

    for eco, r in results.items():
        eco_cfg = config["ecosystems"][eco.lower()]
        report["ecosystems"][eco] = {
            "hurricane":               eco_cfg["hurricane"],
            "hurricane_year":          eco_cfg["hurricane_year"],
            "impacted_area_ha_max":    r.get("impacted_ha_max", 0),
            "impacted_area_ha_mean":   r.get("impacted_ha_mean", 0),
            "carbon_loss_tco2_eq":     r.get("carbon_loss_tco2_eq", 0),
            "recovery_rate_per_quarter": r.get("gap_closing_rate", float("nan")),
            "time_to_convergence_quarters": r.get("time_to_convergence_quarters"),
            "actual_ndvi_slope":       r.get("actual_ndvi_slope", float("nan")),
            "delta_ndvi_first_quarter": (
                r.get("delta_ndvi_mean_timeseries", [float("nan")])[0]
                if r.get("delta_ndvi_mean_timeseries") else float("nan")
            ),
        }

    report_path = metrics_dir / f"consolidated_report_{model_arch}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: None if x != x else x)
    logger.info(f"\nConsolidated report → {report_path}")

    logger.info("\n" + "=" * 50)
    logger.info(f"  KEY FINDINGS — {model_arch}")
    logger.info("=" * 50)
    for eco, data in report["ecosystems"].items():
        logger.info(f"\n  {eco.upper()} — Hurricane {data['hurricane']} {data['hurricane_year']}")
        logger.info(f"    Max impacted area:   {data['impacted_area_ha_max']:.1f} ha")
        logger.info(f"    Carbon loss:         {data['carbon_loss_tco2_eq']:.1f} tCO₂-eq")
        ttc = data.get("time_to_convergence_quarters")
        if ttc is not None:
            logger.info(f"    Time to recovery:    {ttc} quarters (~{ttc/4:.1f} years)")
        else:
            logger.info(f"    Time to recovery:    Not reached in observation window")
        recovery = data.get("recovery_rate_per_quarter", float("nan"))
        if recovery == recovery:
            logger.info(f"    Gap closing rate:    {recovery:.4f} NDVI/quarter")
    logger.info("=" * 50)


def _save_comparison_csv(results: dict, metrics_dir: Path, model_arch: str):
    rows = []
    for eco, r in results.items():
        rows.append({
            "model":              model_arch,
            "ecosystem":          eco,
            "impacted_ha_max":    r.get("impacted_ha_max",      float("nan")),
            "impacted_ha_mean":   r.get("impacted_ha_mean",     float("nan")),
            "carbon_loss_tco2_eq": r.get("carbon_loss_tco2_eq", float("nan")),
            "actual_ndvi_slope":  r.get("actual_ndvi_slope",    float("nan")),
            "cf_ndvi_slope":      r.get("cf_ndvi_slope",        float("nan")),
            "gap_closing_rate":   r.get("gap_closing_rate",     float("nan")),
            "time_to_convergence_q": r.get("time_to_convergence_quarters"),
            "n_post_quarters":    r.get("n_post_quarters", 0),
        })
    df = pd.DataFrame(rows)
    csv_path = metrics_dir / f"comparison_table_{model_arch}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison table → {csv_path}")
    logger.info(f"\n{df.to_string()}")


def _save_cross_model_comparison(all_model_results: dict, metrics_dir: Path, ecosystems: list):
    """Build a single table comparing all models side by side."""
    rows = []
    for model_arch, eco_results in all_model_results.items():
        for eco, r in eco_results.items():
            rows.append({
                "model":              model_arch,
                "ecosystem":          eco,
                "impacted_ha_max":    r.get("impacted_ha_max",       float("nan")),
                "carbon_loss_tco2_eq": r.get("carbon_loss_tco2_eq",  float("nan")),
                "gap_closing_rate":   r.get("gap_closing_rate",      float("nan")),
                "time_to_convergence_q": r.get("time_to_convergence_quarters"),
                "actual_ndvi_slope":  r.get("actual_ndvi_slope",     float("nan")),
            })
    df = pd.DataFrame(rows)
    csv_path = metrics_dir / "cross_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nCross-model comparison → {csv_path}")
    logger.info(f"\n{df.to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and consolidate metrics")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=ALL_MODELS,
        help="Model to compute metrics for (default: from config.yaml)",
    )
    parser.add_argument(
        "--all-models", action="store_true", dest="all_models",
        help="Auto-detect and process all models that have generated arrays",
    )
    parser.add_argument(
        "--ecosystem", type=str, default=None,
        choices=["everglades", "mississippi"],
    )
    main(parser.parse_args())