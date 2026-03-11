"""
01_build_tensors.py
-------------------
Step 1: Load all GeoTIFF composites from Google Drive (or local),
        stack into (T, C, H, W) tensors, fit normalizers, extract patches.
 
Run this ONCE before any training.
 
Usage:
    python scripts/01_build_tensors.py
    python scripts/01_build_tensors.py --force   # rebuild even if cached
    python scripts/01_build_tensors.py --ecosystem everglades
"""
 
import sys
import os
import argparse
import logging
 
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import yaml
import numpy as np
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
 
 
def main(args):
    # --- Load config ---
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
 
    logger.info("="*60)
    logger.info(" ECO-REWIND: Step 1 — Build Tensors & Patches")
    logger.info("="*60)
    logger.info(f"  Data mode:     {config['data']['mode']}")
    logger.info(f"  Processed dir: {config['data']['processed_dir']}")
    logger.info(f"  Force rebuild: {args.force}")
 
    from data.preprocessing.tensor_builder import TensorBuilder
    from data.preprocessing.normalizer import build_normalizers
    from data.preprocessing.patch_sampler import PatchSampler
 
    builder = TensorBuilder(config)
    sampler = PatchSampler(config)
 
    ecosystems = (
        [args.ecosystem]
        if args.ecosystem
        else list(config["ecosystems"].keys())
    )
 
    tensors = {}
    validities = {}
 
    # --- Step 1: Build tensors ---
    logger.info("\n--- Building tensors ---")
    for eco in ecosystems:
        logger.info(f"\n[{eco.upper()}]")
        result = builder.build(eco, force_rebuild=args.force)
        tensors[eco] = result["tensor"]
        validities[eco] = result["validity"]
 
    # --- Step 2: Fit normalizers ---
    logger.info("\n--- Fitting normalizers (pre-disturbance only) ---")
    normalizers = build_normalizers(
        config, tensors,
        processed_dir=config["data"]["processed_dir"],
        force_refit=args.force,
    )
 
    # --- Step 3: Normalize tensors ---
    logger.info("\n--- Normalizing tensors ---")
    norm_tensors = {}
    for eco, tensor in tensors.items():
        logger.info(f"  Normalizing {eco}...")
        norm_tensors[eco] = normalizers[eco].transform(tensor)
        # Save normalized tensor
        norm_path = Path(config["data"]["processed_dir"]) / eco / "tensor_normalized.npy"
        np.save(str(norm_path), norm_tensors[eco])
        logger.info(f"  Saved normalized tensor → {norm_path}")
 
    # --- Step 4: Extract patches ---
    logger.info("\n--- Extracting patches ---")
    for eco, norm_tensor in norm_tensors.items():
        logger.info(f"\n[{eco.upper()}] Extracting patches...")
        patch_dir = sampler.build_patches(
            ecosystem=eco,
            tensor=norm_tensor,
            validity=validities[eco],
            normalizer=normalizers[eco],
            force_rebuild=args.force,
        )
        logger.info(f"  Patches saved → {patch_dir}")
 
    # --- Summary ---
    logger.info("\n" + "="*60)
    logger.info(" Build Complete!")
    logger.info("="*60)
    for eco in ecosystems:
        patch_file = Path(config["data"]["processed_dir"]) / eco / "patches" / "patches.npy"
        if patch_file.exists():
            shape = np.load(str(patch_file), mmap_mode="r").shape
            logger.info(f"  {eco}: {shape[0]} patches, shape={shape}")
 
    logger.info("\nNext step: python scripts/02_train_convlstm.py")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build tensors and patches")
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild even if cached files exist"
    )
    parser.add_argument(
        "--ecosystem", type=str, default=None,
        choices=["everglades", "mississippi"],
        help="Build only one ecosystem (default: both)"
    )
    main(parser.parse_args())