"""
05_compute_metrics.py
---------------------
Step 5: Load saved counterfactual arrays and compute all ecological
        metrics, then generate a consolidated report.
 
Run after 04_generate_counterfactual.py.
Can also be run standalone if .npy arrays already exist.
 
Outputs:
  - outputs/metrics/consolidated_report.json
  - outputs/metrics/comparison_table.csv  (for both ecosystems side by side)
  - outputs/maps/ablation_comparison.png  (if multiple models run)
 
Usage:
    python scripts/05_compute_metrics.py
    python scripts/05_compute_metrics.py --ecosystem everglades
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
 
 
def main(args):
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
 
    logger.info("="*60)
    logger.info(" ECO-REWIND: Step 5 — Compute & Consolidate Metrics")
    logger.info("="*60)
 
    processed_dir = Path(config["data"]["processed_dir"])
    metrics_dir = Path(config["outputs"]["metrics"])
    maps_dir = Path(config["outputs"]["maps"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
 
    ecosystems = (
        [args.ecosystem] if args.ecosystem
        else list(config["ecosystems"].keys())
    )
 
    from inference.metrics import EcologicalMetrics
    from visualization.maps import plot_trajectories, plot_recovery_metrics
 
    metrics_engine = EcologicalMetrics(config)
    all_results = {}
 
    for eco in ecosystems:
        logger.info(f"\n--- {eco.upper()} ---")
 
        # Check if pre-generated .npy files exist
        delta_path = processed_dir / eco / "delta.npy"
        cf_path = processed_dir / eco / "counterfactual.npy"
        actual_path = processed_dir / eco / "actual_post.npy"
 
        if not delta_path.exists():
            logger.warning(
                f"  No counterfactual arrays found for {eco}.\n"
                f"  Run: python scripts/04_generate_counterfactual.py --ecosystem {eco}"
            )
            continue
 
        delta = np.load(str(delta_path))
        cf = np.load(str(cf_path))
        actual = np.load(str(actual_path))
 
        cf_results = {
            "counterfactual": cf,
            "actual_post": actual,
            "delta": delta,
        }
 
        logger.info(f"  Loaded arrays: delta={delta.shape}, cf={cf.shape}, actual={actual.shape}")
 
        # Compute metrics
        results = metrics_engine.compute_all(eco, cf_results, pixel_size_m=10.0)
        all_results[eco] = results
 
        # Regenerate trajectory plots (in case visualizations were not generated)
        plot_trajectories(results, config, eco, str(maps_dir))
        plot_recovery_metrics(results, config, eco, str(maps_dir))
 
        # Save per-ecosystem metrics
        eco_metrics_path = metrics_dir / f"{eco}_metrics_final.json"
        metrics_engine.save(results, str(eco_metrics_path))
 
    # --- Consolidated report ---
    if all_results:
        _save_consolidated_report(all_results, metrics_dir, config)
        _save_comparison_csv(all_results, metrics_dir)
 
    logger.info("\n" + "="*60)
    logger.info(" Metrics computation complete!")
    logger.info(f"  Reports:  {metrics_dir}/")
    logger.info(f"  Maps:     {maps_dir}/")
    logger.info("="*60)
    logger.info("\nNext step: python scripts/06_ablation.py (optional)")
    logger.info("  OR: Prepare results for thesis/paper!")
 
 
def _save_consolidated_report(results: dict, metrics_dir: Path, config: dict):
    """Save a human-readable consolidated JSON report."""
    report = {
        "project": "ECO-REWIND",
        "ecosystems": {},
    }
 
    for eco, r in results.items():
        eco_cfg = config["ecosystems"][eco.lower()]
        report["ecosystems"][eco] = {
            "hurricane": eco_cfg["hurricane"],
            "hurricane_year": eco_cfg["hurricane_year"],
            "impacted_area_ha_max": r.get("impacted_ha_max", 0),
            "impacted_area_ha_mean": r.get("impacted_ha_mean", 0),
            "carbon_loss_tco2_eq": r.get("carbon_loss_tco2_eq", 0),
            "recovery_rate_per_quarter": r.get("gap_closing_rate", float("nan")),
            "time_to_convergence_quarters": r.get("time_to_convergence_quarters"),
            "actual_ndvi_slope": r.get("actual_ndvi_slope", float("nan")),
            "delta_ndvi_mean_first_quarter": (
                r.get("delta_ndvi_mean_timeseries", [float("nan")])[0]
                if r.get("delta_ndvi_mean_timeseries") else float("nan")
            ),
        }
 
    report_path = metrics_dir / "consolidated_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: None if x != x else x)
    logger.info(f"\nConsolidated report → {report_path}")
 
    # Pretty-print key findings
    logger.info("\n" + "="*50)
    logger.info("  KEY FINDINGS")
    logger.info("="*50)
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
        if recovery == recovery:  # not nan
            logger.info(f"    Gap closing rate:    {recovery:.4f} NDVI/quarter")
    logger.info("="*50)
 
 
def _save_comparison_csv(results: dict, metrics_dir: Path):
    """Save a comparison table CSV for both ecosystems."""
    rows = []
    for eco, r in results.items():
        rows.append({
            "ecosystem": eco,
            "hurricane": r.get("ecosystem", eco),
            "impacted_ha_max": r.get("impacted_ha_max", float("nan")),
            "impacted_ha_mean": r.get("impacted_ha_mean", float("nan")),
            "carbon_loss_tco2_eq": r.get("carbon_loss_tco2_eq", float("nan")),
            "actual_ndvi_slope": r.get("actual_ndvi_slope", float("nan")),
            "cf_ndvi_slope": r.get("cf_ndvi_slope", float("nan")),
            "gap_closing_rate": r.get("gap_closing_rate", float("nan")),
            "time_to_convergence_q": r.get("time_to_convergence_quarters"),
            "n_post_quarters": r.get("n_post_quarters", 0),
        })
 
    df = pd.DataFrame(rows)
    csv_path = metrics_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison table → {csv_path}")
    logger.info(f"\n{df.to_string()}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and consolidate metrics")
    parser.add_argument("--ecosystem", type=str, default=None,
                        choices=["everglades", "mississippi"])
    main(parser.parse_args())