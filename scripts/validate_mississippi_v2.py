"""
validate_mississippi_v2.py
--------------------------
Local validation + uint16 conversion for Mississippi v2 GeoTIFFs.
 
Run this BEFORE building tensors to confirm the Barataria Bay re-export
is healthy and usable. Then it converts float32 GEE exports to uint16.
 
Usage:
    python scripts/validate_mississippi_v2.py
    python scripts/validate_mississippi_v2.py --skip-convert   (validate only)
    python scripts/validate_mississippi_v2.py --delete-originals  (free space after conversion)
 
What "good results" look like:
    NDVI (pre-disturbance, 2016–2020) : 0.20 – 0.55  (Spartina marsh)
    NDVI (2021-Q3, Ida impact)        : drop ≥ 0.05 vs prior quarter
    NDVI (2022–2023, recovery)        : trending back toward pre-disturbance
    SAR_VV                            : −25 to 0 dB  (wetland range)
    NaN coverage per quarter          : < 60%  (cloud gaps are OK for median composites)
 
    FAIL indicators (v1 was wrong AOI):
    NDVI ≈ −0.02 to +0.05 → open water / mud flat (wrong AOI, same problem as v1)
    NaN coverage > 90%     → quarter entirely clouded out, needs relaxed cloud filter
"""
 
import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
 
import numpy as np
 
try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. Run:  pip install rasterio")
    sys.exit(1)
 
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("NOTE: matplotlib not found — skipping plots. Install with: pip install matplotlib")
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
V2_DIR   = Path("/home/CL502-17/Desktop/Kunal_IDL_V1/EcoRewind_Data/Mississippi_v2")
OUT_DIR  = V2_DIR   # uint16 files written alongside originals
 
# Band order in the GeoTIFFs (must match GEE export order)
BAND_NAMES = ["Blue", "Green", "Red", "NIR", "NDVI", "NDWI", "SAR_VV"]
BAND_IDX   = {name: i for i, name in enumerate(BAND_NAMES)}
 
# uint16 encoding — identical to Colab Cell 7 so downstream pipeline is compatible
UINT16_SCALE = {
    "Blue":   {"offset": 0.0,  "mult": 10_000, "clip": (0, 10_000)},
    "Green":  {"offset": 0.0,  "mult": 10_000, "clip": (0, 10_000)},
    "Red":    {"offset": 0.0,  "mult": 10_000, "clip": (0, 10_000)},
    "NIR":    {"offset": 0.0,  "mult": 10_000, "clip": (0, 10_000)},
    "NDVI":   {"offset": 1.0,  "mult": 10_000, "clip": (0, 20_000)},
    "NDWI":   {"offset": 1.0,  "mult": 10_000, "clip": (0, 20_000)},
    "SAR_VV": {"offset": 25.0, "mult":  1_000, "clip": (0, 25_000)},
}
 
# Validation thresholds for Barataria Bay Spartina marsh
THRESHOLDS = {
    "ndvi_min":       0.10,   # absolute floor (even winter/post-storm)
    "ndvi_max":       0.65,   # absolute ceiling (dense summer marsh)
    "ndvi_open_water": 0.08,  # if mean NDVI < this → likely wrong AOI
    "nan_pct_warn":   0.60,   # warn if >60% of pixels are NaN (cloud gap)
    "nan_pct_fail":   0.90,   # fail if >90% NaN (quarter unusable)
    "sar_vv_min":    -35.0,   # dB floor (anything below is noise)
    "sar_vv_max":     5.0,    # dB ceiling
}
 
# Disturbance event
IDA_YEAR    = 2021
IDA_QUARTER = 3   # Q3 = Jul-Sep (Ida made landfall Aug 29)
 
QUARTER_LABELS = {1: "Q1_JanMar", 2: "Q2_AprJun", 3: "Q3_JulSep", 4: "Q4_OctDec"}
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def parse_filename(fname: str):
    """
    Extract (year, quarter_num) from filenames like:
      Mississippi_v2_2021_Q3_JulSep.tif
      Mississippi_v2_2021_Q3_JulSep_u16.tif
    Returns (year, quarter_num) or (None, None) if parse fails.
    """
    stem = Path(fname).stem.replace("_u16", "")
    parts = stem.split("_")
    try:
        year = int([p for p in parts if p.isdigit() and len(p) == 4][0])
        q_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        q_num = next((q_map[p] for p in parts if p in q_map), None)
        return year, q_num
    except (IndexError, StopIteration, ValueError):
        return None, None
 
 
def read_tif_stats(tif_path: Path) -> dict:
    """
    Read a float32 GeoTIFF and compute per-band statistics.
    Returns dict with means, stds, nan_pct, shape, crs.
    """
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)   # (bands, H, W)
        crs  = str(src.crs)
        res  = src.res
        shape = data.shape
 
    n_bands, H, W = shape
    band_names = BAND_NAMES[:n_bands]
 
    stats = {
        "path":     str(tif_path),
        "shape":    shape,
        "crs":      crs,
        "res_m":    res,
        "bands":    {},
    }
 
    for i, bname in enumerate(band_names):
        band = data[i]
        valid = band[np.isfinite(band)]
        nan_pct = 1.0 - len(valid) / (H * W)
        stats["bands"][bname] = {
            "mean":    float(np.mean(valid)) if len(valid) > 0 else np.nan,
            "std":     float(np.std(valid))  if len(valid) > 0 else np.nan,
            "min":     float(np.min(valid))  if len(valid) > 0 else np.nan,
            "max":     float(np.max(valid))  if len(valid) > 0 else np.nan,
            "nan_pct": float(nan_pct),
        }
 
    return stats
 
 
def assess_quarter(stats: dict, year: int, q_num: int) -> list:
    """
    Return list of (level, message) tuples: level in {OK, WARN, FAIL}.
    """
    issues = []
    bands  = stats["bands"]
 
    if "NDVI" not in bands:
        issues.append(("FAIL", "NDVI band missing entirely"))
        return issues
 
    ndvi  = bands["NDVI"]
    ndwi  = bands.get("NDWI", {})
    sar   = bands.get("SAR_VV", {})
 
    # NaN coverage
    nan_pct = ndvi.get("nan_pct", 1.0)
    if nan_pct > THRESHOLDS["nan_pct_fail"]:
        issues.append(("FAIL", f"NDVI NaN={nan_pct*100:.0f}% — quarter entirely cloud-covered"))
    elif nan_pct > THRESHOLDS["nan_pct_warn"]:
        issues.append(("WARN", f"NDVI NaN={nan_pct*100:.0f}% — heavy cloud cover but usable"))
    else:
        issues.append(("OK",   f"NaN coverage {nan_pct*100:.0f}% — acceptable"))
 
    # NDVI range check
    mean_ndvi = ndvi.get("mean", np.nan)
    if np.isnan(mean_ndvi):
        issues.append(("FAIL", "NDVI mean is NaN — no valid pixels"))
    elif mean_ndvi < THRESHOLDS["ndvi_open_water"]:
        issues.append(("FAIL",
            f"NDVI={mean_ndvi:.3f} < {THRESHOLDS['ndvi_open_water']} "
            f"→ OPEN WATER / WRONG AOI (v1 problem was NDVI≈-0.02)"))
    elif mean_ndvi < THRESHOLDS["ndvi_min"] and not (year == IDA_YEAR and q_num == IDA_QUARTER):
        issues.append(("WARN",
            f"NDVI={mean_ndvi:.3f} below expected floor {THRESHOLDS['ndvi_min']} "
            f"(acceptable only immediately post-Ida)"))
    elif mean_ndvi > THRESHOLDS["ndvi_max"]:
        issues.append(("WARN",
            f"NDVI={mean_ndvi:.3f} unusually high — AOI may include upland forest"))
    else:
        issues.append(("OK", f"NDVI={mean_ndvi:.3f} — in expected Spartina marsh range"))
 
    # SAR_VV sanity check
    if sar and not np.isnan(sar.get("mean", np.nan)):
        sar_mean = sar["mean"]
        if not (THRESHOLDS["sar_vv_min"] <= sar_mean <= THRESHOLDS["sar_vv_max"]):
            issues.append(("WARN",
                f"SAR_VV={sar_mean:.1f} dB outside expected [{THRESHOLDS['sar_vv_min']}, "
                f"{THRESHOLDS['sar_vv_max']}] dB"))
        else:
            issues.append(("OK", f"SAR_VV={sar_mean:.1f} dB — valid wetland range"))
 
    return issues
 
 
def convert_to_uint16(src_path: Path, dst_path: Path):
    """Convert float32 GeoTIFF to uint16 with per-band scaling."""
    with rasterio.open(src_path) as src:
        data    = src.read().astype(np.float32)
        profile = src.profile.copy()
        names   = src.descriptions or BAND_NAMES[:src.count]
 
    n_bands = data.shape[0]
    out     = np.zeros_like(data, dtype=np.uint16)
 
    for i in range(n_bands):
        bname = (names[i] or BAND_NAMES[i]) if i < len(BAND_NAMES) else f"Band{i+1}"
        p = UINT16_SCALE.get(bname, None)
        if p:
            scaled = (data[i] + p["offset"]) * p["mult"]
            out[i] = np.clip(scaled, *p["clip"]).astype(np.uint16)
        else:
            bmin, bmax = data[i].min(), data[i].max()
            rng = bmax - bmin or 1.0
            out[i] = np.clip((data[i] - bmin) / rng * 65535, 0, 65535).astype(np.uint16)
 
    profile.update(dtype="uint16", compress="lzw", predictor=2, nodata=0)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out)
        band_list = names if names else [f"Band{j+1}" for j in range(n_bands)]
        for i, n in enumerate(band_list):
            dst.set_band_description(i + 1, n or f"Band{i+1}")
 
    before_mb = src_path.stat().st_size / 1e6
    after_mb  = dst_path.stat().st_size / 1e6
    return before_mb, after_mb
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main(args):
    print("=" * 70)
    print("  ECO-REWIND: Mississippi v2 Validation + uint16 Conversion")
    print("=" * 70)
    print(f"  Data path : {V2_DIR}")
    print(f"  Expected  : 32 float32 GeoTIFFs  (2016–2023 × 4 quarters)")
    print()
 
    if not V2_DIR.exists():
        print(f"ERROR: Directory not found: {V2_DIR}")
        sys.exit(1)
 
    # ── Find float32 GeoTIFFs ─────────────────────────────────
    tifs = sorted([f for f in V2_DIR.glob("*.tif") if "_u16" not in f.name])
    if not tifs:
        print(f"ERROR: No .tif files found in {V2_DIR}")
        sys.exit(1)
 
    print(f"Found {len(tifs)} float32 GeoTIFF(s)")
    if len(tifs) != 32:
        print(f"  WARNING: Expected 32, got {len(tifs)}. Missing quarters listed at end.")
    print()
 
    # ── Per-file stats ────────────────────────────────────────
    all_stats  = {}   # (year, q_num) → stats dict
    all_issues = {}
    ndvi_series = {}  # (year, q_num) → mean NDVI  (for trend plot)
 
    print(f"{'File':<45}  {'NDVI':>7}  {'NDWI':>7}  {'SAR_VV':>8}  {'NaN%':>5}  Status")
    print("─" * 90)
 
    fail_count = warn_count = ok_count = 0
 
    for tif in tifs:
        year, q_num = parse_filename(tif.name)
        if year is None:
            print(f"  SKIP {tif.name} — cannot parse year/quarter from filename")
            continue
 
        stats  = read_tif_stats(tif)
        issues = assess_quarter(stats, year, q_num)
        all_stats[(year, q_num)]  = stats
        all_issues[(year, q_num)] = issues
 
        bands     = stats["bands"]
        ndvi_mean = bands.get("NDVI", {}).get("mean", np.nan)
        ndwi_mean = bands.get("NDWI", {}).get("mean", np.nan)
        sar_mean  = bands.get("SAR_VV", {}).get("mean", np.nan)
        nan_pct   = bands.get("NDVI", {}).get("nan_pct", 1.0) * 100
 
        levels = [lvl for lvl, _ in issues]
        if "FAIL" in levels:
            status = "❌ FAIL"; fail_count += 1
        elif "WARN" in levels:
            status = "⚠ WARN"; warn_count += 1
        else:
            status = "✅ OK";   ok_count   += 1
 
        ndvi_str  = f"{ndvi_mean:+.3f}" if not np.isnan(ndvi_mean) else "  NaN"
        ndwi_str  = f"{ndwi_mean:+.3f}" if not np.isnan(ndwi_mean) else "  NaN"
        sar_str   = f"{sar_mean:+.1f}"  if not np.isnan(sar_mean)  else "   NaN"
        nan_str   = f"{nan_pct:.0f}%"
 
        label = f"{tif.stem[:43]}"
        print(f"  {label:<43}  {ndvi_str:>7}  {ndwi_str:>7}  {sar_str:>8}  {nan_str:>5}  {status}")
 
        if not np.isnan(ndvi_mean):
            ndvi_series[(year, q_num)] = ndvi_mean
 
    print("─" * 90)
    print(f"  Results: {ok_count} OK  |  {warn_count} WARN  |  {fail_count} FAIL")
    print()
 
    # ── Check for Ida impact signal ────────────────────────────
    print("─" * 70)
    print("  Hurricane Ida Impact Check (2021-Q3 vs surrounding quarters)")
    print("─" * 70)
 
    pre_ndvi  = [v for (y, q), v in ndvi_series.items() if y in (2019, 2020)]
    ida_ndvi  = ndvi_series.get((IDA_YEAR, IDA_QUARTER), None)
    post_ndvi = [v for (y, q), v in ndvi_series.items() if y in (2022, 2023)]
 
    pre_mean  = float(np.mean(pre_ndvi))  if pre_ndvi  else np.nan
    post_mean = float(np.mean(post_ndvi)) if post_ndvi else np.nan
 
    if ida_ndvi is not None and not np.isnan(pre_mean):
        drop = pre_mean - ida_ndvi
        print(f"  Pre-Ida  (2019–2020 mean)  : NDVI = {pre_mean:+.3f}")
        print(f"  Ida impact (2021-Q3)       : NDVI = {ida_ndvi:+.3f}")
        print(f"  Drop                       : {drop:+.3f}")
        if drop > 0.05:
            print(f"  ✅ Clear hurricane impact signal detected (drop={drop:.3f})")
        elif drop > 0.02:
            print(f"  ⚠ Weak signal — drop only {drop:.3f}. May be seasonal, check 2021-Q2.")
        else:
            print(f"  ❌ No detectable Ida signal. NDVI barely changed — check AOI or cloud cover.")
        if not np.isnan(post_mean):
            recovery = post_mean - ida_ndvi
            print(f"  Recovery (2022–2023 mean)  : NDVI = {post_mean:+.3f}  (+{recovery:.3f} from Ida quarter)")
    else:
        print(f"  ⚠ Cannot assess: 2021-Q3 NDVI not available (missing or all-NaN)")
    print()
 
    # ── vs v1 comparison summary ───────────────────────────────
    print("─" * 70)
    print("  Comparison: v2 (Barataria Bay) vs v1 (open delta — wrong AOI)")
    print("─" * 70)
    all_ndvi = [v for v in ndvi_series.values() if not np.isnan(v)]
    if all_ndvi:
        overall_mean = float(np.mean(all_ndvi))
        v1_reference = -0.02   # typical v1 mean NDVI (open water)
        print(f"  v1 mean NDVI (open water)  : {v1_reference:+.3f}  ← this is what we had before")
        print(f"  v2 mean NDVI (all quarters): {overall_mean:+.3f}  ← this is now")
        improvement = overall_mean - v1_reference
        if improvement > 0.15:
            print(f"  ✅ MAJOR improvement: +{improvement:.3f} NDVI units over v1")
            print(f"     v2 data is VALID for counterfactual modeling.")
        elif improvement > 0.05:
            print(f"  ⚠ Moderate improvement: +{improvement:.3f} — check NDVI is in 0.20–0.45 range")
        else:
            print(f"  ❌ Little improvement over v1 — check AOI coordinates, still may be water")
    print()
 
    # ── Missing quarter report ─────────────────────────────────
    all_expected = {(y, q) for y in range(2016, 2024) for q in range(1, 5)}
    found        = set(all_stats.keys())
    missing      = sorted(all_expected - found)
    if missing:
        print("─" * 70)
        print(f"  Missing quarters ({len(missing)}):")
        for y, q in missing:
            print(f"    {y} {QUARTER_LABELS.get(q, f'Q{q}')} — not found in directory")
        print("  These will appear as NaN time steps in the tensor.")
        print("  Up to 8 missing quarters is acceptable for Sentinel-2 (cloud gaps).")
        print()
 
    # ── NDVI time-series plot ──────────────────────────────────
    if HAS_MPL and ndvi_series:
        sorted_keys = sorted(ndvi_series.keys())
        x_vals  = [y + (q - 1) / 4 for y, q in sorted_keys]
        y_vals  = [ndvi_series[(y, q)] for y, q in sorted_keys]
        ida_x   = IDA_YEAR + (IDA_QUARTER - 1) / 4
 
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(x_vals, y_vals, "o-", color="#2ca02c", linewidth=1.8, markersize=5, label="NDVI (Barataria Bay)")
        ax.axvline(ida_x, color="red", linestyle="--", linewidth=1.5, label="Hurricane Ida (Aug 2021)")
        ax.axhline(0.0,   color="gray", linestyle=":", linewidth=0.8)
        ax.axhspan(-0.10, 0.08, alpha=0.12, color="red",   label="Open-water zone (v1 was here)")
        ax.axhspan( 0.20,  0.55, alpha=0.08, color="green", label="Expected Spartina range")
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Mean NDVI", fontsize=12)
        ax.set_title("Mississippi v2 — Barataria Bay NDVI Time Series\n(should show drop at Ida, recovery after)", fontsize=13)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_ylim(-0.15, 0.70)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
 
        plot_path = V2_DIR / "v2_ndvi_timeseries.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  📊 NDVI time-series plot saved → {plot_path}")
        print()
 
    # ── Print per-file issues (only WARN and FAIL) ─────────────
    problem_quarters = [(k, v) for k, v in all_issues.items()
                        if any(lvl in ("WARN", "FAIL") for lvl, _ in v)]
    if problem_quarters:
        print("─" * 70)
        print("  Detailed issues for flagged quarters:")
        for (year, q_num), issues in sorted(problem_quarters):
            print(f"\n  [{year} {QUARTER_LABELS.get(q_num, f'Q{q_num}')}]")
            for level, msg in issues:
                symbol = "  ❌" if level == "FAIL" else "  ⚠"
                print(f"    {symbol} {msg}")
        print()
 
    # ── Conversion ────────────────────────────────────────────
    if args.skip_convert:
        print("─" * 70)
        print("  Skipping uint16 conversion (--skip-convert flag set).")
        print("  To convert, run without the flag.")
    elif fail_count > 0 and not args.force_convert:
        print("─" * 70)
        print(f"  ⚠  {fail_count} FAIL(s) detected. Skipping uint16 conversion.")
        print("  Review the issues above. If they're acceptable (e.g. all-cloud quarters),")
        print("  re-run with --force-convert to convert anyway.")
    else:
        print("─" * 70)
        print("  Converting float32 GeoTIFFs → uint16 (LZW compressed)...")
        print("  (Same encoding as Everglades — compatible with 01_build_tensors.py)")
        print()
 
        total_before = total_after = 0.0
        converted    = 0
 
        for tif in tifs:
            dst = tif.parent / tif.name.replace(".tif", "_u16.tif")
            if dst.exists() and not args.force_convert:
                print(f"  SKIP (already exists): {dst.name}")
                continue
 
            before_mb, after_mb = convert_to_uint16(tif, dst)
            total_before += before_mb
            total_after  += after_mb
            converted    += 1
            pct = 100 * (1 - after_mb / before_mb)
            print(f"  ✅ {tif.name[:50]:50s}  {before_mb:6.1f} → {after_mb:6.1f} MB  ({pct:.0f}% smaller)")
 
            if args.delete_originals:
                tif.unlink()
 
        if converted > 0:
            saved_pct = 100 * (1 - total_after / total_before)
            print()
            print(f"  Converted : {converted} files")
            print(f"  Before    : {total_before:.1f} MB  ({total_before/1024:.2f} GB)")
            print(f"  After     : {total_after:.1f} MB  ({total_after/1024:.2f} GB)")
            print(f"  Saved     : {total_before - total_after:.1f} MB  ({saved_pct:.0f}% reduction)")
        else:
            print("  All files already converted.")
 
    # ── Final verdict ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)
 
    all_ndvi_vals = [v for v in ndvi_series.values() if not np.isnan(v)]
    overall       = float(np.mean(all_ndvi_vals)) if all_ndvi_vals else np.nan
 
    if fail_count == 0 and not np.isnan(overall) and overall > 0.15:
        print(f"  ✅ DATA IS VALID — proceed to tensor building")
        print(f"     Mean NDVI = {overall:.3f}  (Spartina marsh range confirmed)")
        print()
        print("  Next steps:")
        print("    1. python scripts/01_build_tensors.py --ecosystem mississippi")
        print("    2. In configs/config.yaml set:")
        print("         mississippi:")
        print("           version: \"v2\"")
        print("           folder:  \"Mississippi_v2\"")
        print("         ecosystems_for_training:")
        print("           mississippi: true")
        print("    3. python scripts/02_train_convlstm.py")
    elif fail_count > 0:
        print(f"  ❌ {fail_count} quarter(s) FAILED validation")
        print(f"     Check issues above. If NDVI is near 0 → AOI is wrong location.")
        print(f"     If NDVI is in range but NaN is high → cloud cover, acceptable.")
    else:
        print(f"  ⚠  Data partially valid (mean NDVI={overall:.3f}). Review WARN items above.")
 
    print("=" * 70)
 
    # ── Save validation report ─────────────────────────────────
    report = {
        "generated_at":    datetime.now().isoformat(),
        "data_path":       str(V2_DIR),
        "files_found":     len(tifs),
        "missing_quarters": [f"{y}-{QUARTER_LABELS.get(q, f'Q{q}')}" for y, q in missing],
        "summary": {
            "ok":    ok_count,
            "warn":  warn_count,
            "fail":  fail_count,
            "overall_mean_ndvi":  round(overall, 4) if not np.isnan(overall) else None,
            "ida_ndvi":           round(ida_ndvi, 4) if ida_ndvi and not np.isnan(ida_ndvi) else None,
            "pre_ida_mean_ndvi":  round(pre_mean, 4) if not np.isnan(pre_mean) else None,
            "post_ida_mean_ndvi": round(post_mean, 4) if not np.isnan(post_mean) else None,
        },
        "per_quarter": {
            f"{y}_{QUARTER_LABELS.get(q, f'Q{q}')}": {
                "ndvi": round(ndvi_series.get((y, q), np.nan), 4)
                        if not np.isnan(ndvi_series.get((y, q), np.nan)) else None,
                "issues": [(lvl, msg) for lvl, msg in all_issues.get((y, q), [])],
            }
            for y in range(2016, 2024) for q in range(1, 5)
            if (y, q) in all_stats
        },
    }
    report_path = V2_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report → {report_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Mississippi v2 GeoTIFFs and convert to uint16")
    parser.add_argument("--skip-convert",     action="store_true", help="Validate only, do not convert")
    parser.add_argument("--force-convert",    action="store_true", help="Convert even if FAILs exist")
    parser.add_argument("--delete-originals", action="store_true", help="Delete float32 files after conversion")
    main(parser.parse_args())