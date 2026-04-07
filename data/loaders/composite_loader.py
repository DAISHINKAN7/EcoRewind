"""
composite_loader.py
-------------------
Smart loader that reads quarterly GeoTIFF composites from either:
  - Google Drive (mode="gdrive")  — Colab or rclone-mounted RunPod
  - Local filesystem (mode="local")
 
Returns a list of CompositeFile objects sorted chronologically.
"""
 
import os
import re
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
 
import numpy as np
import rasterio
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
 
@dataclass
class CompositeFile:
    path: str
    ecosystem: str        # "everglades" or "mississippi"
    year: int
    quarter: int          # 1–4
    time_index: int       # 0-indexed position in sorted sequence
    is_interpolated: bool = False
    is_mask: bool = False
    quality: str = "GOOD" # GOOD / FAIR / POOR / UNKNOWN
 
    @property
    def label(self) -> str:
        return f"{self.ecosystem}_{self.year}_Q{self.quarter}"
 
    @property
    def sort_key(self) -> int:
        return self.year * 10 + self.quarter
 
 
@dataclass
class EcosystemBundle:
    ecosystem: str
    composites: List[CompositeFile]
    masks: List[CompositeFile]
    metadata: Dict[str, Any] = field(default_factory=dict)
 
    @property
    def n_composites(self) -> int:
        return len(self.composites)
 
 
# ---------------------------------------------------------------------------
# Drive mounting
# ---------------------------------------------------------------------------
 
def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """
    Attempts to mount Google Drive. Works in Colab automatically.
    On RunPod, expects rclone to already be mounted at the given mount_point.
    Returns True if mount point exists and is accessible.
    """
    if os.path.isdir(mount_point) and os.listdir(mount_point):
        logger.info(f"Drive already mounted at {mount_point}")
        return True
 
    # Try Colab mounting
    try:
        from google.colab import drive
        drive.mount(mount_point)
        logger.info(f"Google Drive mounted at {mount_point}")
        return True
    except (ImportError, Exception) as e:
        logger.warning(f"Colab Drive mount failed: {e}")
 
    # Check if rclone has mounted it
    if os.path.isdir(mount_point):
        logger.info(f"Drive mount point exists at {mount_point} (rclone?)")
        return True
 
    logger.error(
        f"Could not mount Google Drive at {mount_point}.\n"
        "On RunPod, run:\n"
        "  mkdir -p /content/drive/MyDrive\n"
        "  rclone mount gdrive: /content/drive/MyDrive --daemon"
    )
    return False
 
 
# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------
 
# Regex patterns for common GEE export naming conventions
# Matches: Everglades_2017_Q3.tif, everglades_2017_q3.tif,
#          Mississippi_2021_Q3_interpolated.tif, Everglades_2016_Q3_mask.tif
_FILENAME_PATTERNS = [
    # Also handles month ranges and multiple suffixes:
    #   Everglades_2017_Q2_AprJun_INTERPOLATED.tif
    #   Everglades_2016_Q3_JulSep_mask.tif
    re.compile(
        r"(?P<eco>[A-Za-z]+)_(?P<year>\d{4})_[Qq](?P<quarter>[1-4])"
        r"(?P<rest>[^.]*)?\.tif$",
        re.IGNORECASE,
    ),
    # With version: Mississippi_v2_2021_Q3.tif
    re.compile(
        r"(?P<eco>[A-Za-z]+)_v\d+_(?P<year>\d{4})_[Qq](?P<quarter>[1-4])"
        r"(?P<rest>[^.]*)?\.tif$",
        re.IGNORECASE,
    ),
]
 
 
def _parse_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse year/quarter from a GeoTIFF filename.
    Returns dict with keys: eco, year, quarter, is_interpolated, is_mask
    Returns None if parsing fails.
    Handles month ranges in filenames: Everglades_2017_Q2_AprJun_INTERPOLATED.tif
    """
    basename = os.path.basename(filename)
    for pattern in _FILENAME_PATTERNS:
        m = pattern.match(basename)
        if m:
            rest = (m.group("rest") or "").lower()   # e.g. "_aprjun_interpolated"
            return {
                "eco": m.group("eco").lower(),
                "year": int(m.group("year")),
                "quarter": int(m.group("quarter")),
                "is_interpolated": "interp" in rest or "fill" in rest,
                "is_mask": "mask" in rest,
            }
    return None
 
 
# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------
 
class CompositeLoader:
    """
    Discovers and loads all quarterly composites for a given ecosystem.
 
    Usage:
        loader = CompositeLoader(config)
        bundle = loader.load_ecosystem("everglades")
        # bundle.composites → sorted list of CompositeFile objects
    """
 
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config["data"]["mode"]
        self.root = self._resolve_root()
        self._metadata = self._load_metadata()
 
    def _resolve_root(self) -> Path:
        if self.mode == "gdrive":
            gdrive_root = self.config["data"]["gdrive_root"]
            mount_point = str(Path(gdrive_root).parts[0] + "/" + Path(gdrive_root).parts[1])
            mount_google_drive(mount_point)
            root = Path(gdrive_root)
        elif self.mode == "local":
            root = Path(self.config["data"]["local_root"])
        else:
            raise ValueError(f"Unknown data mode: {self.mode}. Use 'gdrive' or 'local'.")
 
        if not root.exists():
            raise FileNotFoundError(
                f"Data root not found: {root}\n"
                f"  mode='{self.mode}'\n"
                f"  Check your configs/config.yaml paths."
            )
        logger.info(f"Data root resolved: {root}")
        return root
 
    def _load_metadata(self) -> Dict[str, Any]:
        meta_file = self.config["data"].get("metadata_file", "pipeline_metadata_v2.json")
        meta_path = self.root / meta_file
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            logger.info(f"Loaded pipeline metadata from {meta_path}")
            return meta
        logger.warning(f"Metadata file not found: {meta_path} — continuing without it")
        return {}
 
    def load_ecosystem(self, ecosystem: str) -> EcosystemBundle:
        """
        Scan the ecosystem folder, parse all TIF files, and return a bundle.
        """
        eco_cfg = self.config["ecosystems"][ecosystem.lower()]
        folder = self.root / eco_cfg["folder"]
 
        if not folder.exists():
            raise FileNotFoundError(
                f"Ecosystem folder not found: {folder}\n"
                f"  Expected: {eco_cfg['folder']}/ inside {self.root}"
            )
 
        # Collect quality info from config
        poor_set = {
            (q["year"], q["quarter"])
            for q in eco_cfg.get("poor_quarters", [])
        }
        fair_set = {
            (q["year"], q["quarter"])
            for q in eco_cfg.get("fair_quarters", [])
        }
        interp_set = {
            (q["year"], q["quarter"])
            for q in eco_cfg.get("interpolated_quarters", [])
        }
 
        composites = []
        masks = []
        unrecognized = []
 
        for fpath in sorted(folder.glob("*.tif")):
            # Skip uint16-encoded copies — tensor builder reads float32 originals only.
            # _u16.tif files use per-band integer encoding (Blue×10000, NDVI×10000+10000,
            # SAR×1000+25000) that must be decoded before use. The float32 GeoTIFFs
            # exported directly from GEE are used as-is without any decoding step.
            if "_u16" in fpath.name:
                continue
 
            parsed = _parse_filename(fpath.name)
            if parsed is None:
                unrecognized.append(fpath.name)
                logger.warning(f"Could not parse filename: {fpath.name}")
                continue
 
            year, quarter = parsed["year"], parsed["quarter"]
            key = (year, quarter)
 
            quality = "GOOD"
            if key in poor_set:
                quality = "POOR"
            elif key in fair_set:
                quality = "FAIR"
            if key in interp_set:
                parsed["is_interpolated"] = True
 
            cf = CompositeFile(
                path=str(fpath),
                ecosystem=ecosystem.lower(),
                year=year,
                quarter=quarter,
                time_index=-1,        # assigned below after sorting
                is_interpolated=parsed["is_interpolated"],
                is_mask=parsed["is_mask"],
                quality=quality,
            )
 
            if cf.is_mask:
                masks.append(cf)
            else:
                composites.append(cf)
 
        # Sort chronologically
        composites.sort(key=lambda c: c.sort_key)
        masks.sort(key=lambda c: c.sort_key)
 
        # Assign time indices based on CALENDAR POSITION, not file order.
        # Sequential assignment shifts all indices when a quarter is missing —
        # e.g. missing 2016_Q2 makes 2021_Q4 land at index 22 instead of 23,
        # breaking t_event alignment.  Calendar-based assignment ensures a
        # missing quarter becomes a NaN placeholder at the correct slot.
        start_year    = eco_cfg.get("start_year", composites[0].year if composites else 2016)
        start_quarter = eco_cfg.get("start_quarter", 1)
        for cf in composites:
            cf.time_index = (cf.year - start_year) * 4 + (cf.quarter - start_quarter)
 
        # Validate count
        expected = eco_cfg["n_composites"]
        if len(composites) != expected:
            logger.warning(
                f"[{ecosystem}] Expected {expected} composites, found {len(composites)}.\n"
                f"  Missing quarters will appear as NaN in tensors.\n"
                f"  This is expected if Mississippi v2 re-export is pending."
            )
        else:
            logger.info(f"[{ecosystem}] Found all {len(composites)} composites.")
 
        if unrecognized:
            logger.warning(
                f"[{ecosystem}] {len(unrecognized)} unrecognized files skipped: {unrecognized[:5]}"
            )
 
        # Print time-index → date mapping for t_event verification
        self._print_index_map(ecosystem, composites, eco_cfg)
 
        # Mississippi v1 AOI warning
        if ecosystem.lower() == "mississippi":
            version = eco_cfg.get("version", "v1")
            if version == "v1":
                logger.warning(
                    "\n" + "="*60 +
                    "\n[MISSISSIPPI] version='v1' — original open-water AOI.\n"
                    "  Expected: NDVI ≈ 0.003 (open water delta — WRONG AOI)\n"
                    "  This data will be loaded but may not show hurricane signal.\n"
                    "  When Barataria Bay re-export is complete:\n"
                    "    1. Place v2 files in Mississippi/ folder\n"
                    "    2. Set version: 'v2' in configs/config.yaml\n" +
                    "="*60
                )
 
        return EcosystemBundle(
            ecosystem=ecosystem.lower(),
            composites=composites,
            masks=masks,
            metadata=self._metadata,
        )
 
    def _print_index_map(
        self, ecosystem: str, composites: List[CompositeFile], eco_cfg: Dict
    ) -> None:
        t_event = eco_cfg["t_event"]
        logger.info(f"\n[{ecosystem.upper()}] Time-index → Date mapping:")
        logger.info(f"{'Index':>6}  {'Label':>20}  {'Quality':>8}  {'Note'}")
        logger.info("-" * 56)
        for cf in composites:
            note = ""
            if cf.time_index == t_event:
                note = f"<-- t_event ({eco_cfg['hurricane']})"
            elif cf.time_index == t_event - 1:
                note = "<-- last pre-event"
            if cf.is_interpolated:
                note += " [INTERPOLATED]"
            logger.info(
                f"{cf.time_index:>6}  {cf.label:>20}  {cf.quality:>8}  {note}"
            )
        logger.info("")
        logger.info(
            f"  t_event={t_event} → {composites[t_event].label if t_event < len(composites) else 'OUT OF RANGE'}\n"
            f"  If this does not match {eco_cfg['hurricane']} (Q{eco_cfg['hurricane_quarter']} "
            f"{eco_cfg['hurricane_year']}), edit t_event in configs/config.yaml."
        )
 
 
    def read_tif(self, path: str) -> np.ndarray:
        """
        Read a GeoTIFF and return a (C, H, W) float32 numpy array.
        NaN values preserved.
        """
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            # rasterio uses nodata — convert to NaN
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
        return data
 
    def read_composite(self, cf: CompositeFile) -> np.ndarray:
        """Read a CompositeFile and return (C, H, W) array."""
        return self.read_tif(cf.path)