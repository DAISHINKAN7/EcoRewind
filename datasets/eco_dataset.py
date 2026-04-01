"""
eco_dataset.py
--------------
PyTorch Dataset and DataLoader factory for ECO-REWIND.
 
Each sample is:
  input  : (T_in, C, H, W)  — T_in consecutive normalized quarters
  target : (T_out, C, H, W) — the T_out quarters immediately following
 
Augmentations (training only):
  - Random horizontal / vertical flips
  - Random 90° rotation
  - Mild per-band intensity jitter (reflectance bands only)
  - Gaussian noise on SAR channel
  All transforms applied CONSISTENTLY across all T timesteps.
"""
 
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
 
class TemporalPatchAugmentation:
    """
    Spatial and spectral augmentations applied identically across all T frames.
 
    CRITICAL: every transform must be spatially consistent — the same operation
    applied to ALL timesteps. Independently flipping different timesteps would
    corrupt temporal relationships and make training impossible.
    """
 
    def __init__(self, config: Dict[str, Any]):
        self.n_bands = config["bands"]["count"]
        self.sar_idx = config["bands"]["indices"]["sar_vv"]
 
    def __call__(self, window: np.ndarray) -> np.ndarray:
        """
        window: (T, C, H, W) float32
        Returns augmented (T, C, H, W)
        """
        # 1. Random horizontal flip (p=0.5)
        if np.random.rand() < 0.5:
            window = window[:, :, :, ::-1].copy()
 
        # 2. Random vertical flip (p=0.5)
        if np.random.rand() < 0.5:
            window = window[:, :, ::-1, :].copy()
 
        # 3. Random 90/180/270° rotation (p=0.5)
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            window = np.rot90(window, k=k, axes=(2, 3)).copy()
 
        # 4. Per-band intensity jitter on reflectance+index bands (p=0.3)
        # Simulates atmospheric variation between acquisition dates.
        if np.random.rand() < 0.3:
            for c in range(self.n_bands - 1):   # skip last band (SAR)
                scale = np.random.uniform(0.95, 1.05)
                window[:, c] = np.clip(window[:, c] * scale, 0.0, 1.0)
 
        # 5. Gaussian noise on SAR channel (p=0.5)
        # SAR has intrinsic speckle — training with noise makes encoder
        # learn speckle-invariant structural features.
        if np.random.rand() < 0.5:
            noise = np.random.normal(0.0, 0.05, window[:, self.sar_idx].shape)
            window[:, self.sar_idx] = (window[:, self.sar_idx] + noise).astype(np.float32)
 
        return window
 
 
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
 
class EcoTimeSeriesDataset(Dataset):
    """
    Sliding-window dataset over ecosystem patches.
 
    Parameters
    ----------
    patches_file : path to (N, T, C, H, W) memory-mapped .npy
    windows      : list of (t_start, t_end) tuples
    t_input      : number of input timesteps
    t_output     : number of target timesteps
    ecosystem    : ecosystem name (for logging)
    augment      : apply TemporalPatchAugmentation (training only)
    config       : full config dict (needed for augmentation)
    """
 
    def __init__(
        self,
        patches_file: str,
        windows:      List[Tuple[int, int]],
        t_input:      int,
        t_output:     int,
        ecosystem:    str = "unknown",
        augment:      bool = False,
        config:       Optional[Dict[str, Any]] = None,
    ):
        self.ecosystem = ecosystem
        self.t_input   = t_input
        self.t_output  = t_output
        self.augment   = augment and (config is not None)
 
        # Load patches as memory-mapped array (no RAM copy)
        self.patches = np.load(patches_file, mmap_mode="r")
        self.N, self.T, self.C, self.H, self.W = self.patches.shape
 
        # Filter windows to those that fit within the tensor
        self.windows = [
            (ts, te) for ts, te in windows
            if te <= self.T and (te - ts) == (t_input + t_output)
        ]
 
        self.augmentation = TemporalPatchAugmentation(config) if self.augment else None
 
        logger.info(
            f"[{ecosystem}] Dataset: {self.N} patches × {len(self.windows)} windows "
            f"= {self.__len__()} samples | C={self.C} H={self.H} W={self.W}"
            f" | augment={self.augment}"
        )
 
    def __len__(self) -> int:
        return self.N * len(self.windows)
 
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch_idx  = idx // len(self.windows)
        window_idx = idx %  len(self.windows)
        t_start, t_end = self.windows[window_idx]
 
        # Slice: (T_in + T_out, C, H, W)
        window_data = self.patches[patch_idx, t_start:t_end].copy()
 
        # Apply augmentation (training only, spatially consistent across T)
        if self.augmentation is not None:
            window_data = self.augmentation(window_data)
 
        inp    = window_data[:self.t_input]    # (T_in, C, H, W)
        target = window_data[self.t_input:]    # (T_out, C, H, W)
 
        return {
            "input":     torch.from_numpy(inp).float(),
            "target":    torch.from_numpy(target).float(),
            "patch_idx": torch.tensor(patch_idx, dtype=torch.long),
            "t_start":   torch.tensor(t_start,   dtype=torch.long),
            "ecosystem": self.ecosystem,
        }
 
 
class CounterfactualDataset(Dataset):
    """
    Dataset for counterfactual inference — feeds pre-disturbance context
    and asks the model to predict forward past the hurricane event.
    No augmentation is applied here.
    """
 
    def __init__(
        self,
        patches_file: str,
        t_event:      int,
        t_input:      int,
        n_forecast:   int,
        ecosystem:    str = "unknown",
    ):
        self.ecosystem  = ecosystem
        self.t_event    = t_event
        self.t_input    = t_input
        self.n_forecast = n_forecast
 
        self.patches = np.load(patches_file, mmap_mode="r")
        self.N, self.T, self.C, self.H, self.W = self.patches.shape
 
        self.input_start = max(0, t_event - t_input)
        self.input_end   = t_event
        self.post_start  = t_event
        self.post_end    = min(self.T, t_event + n_forecast)
 
        logger.info(
            f"[{ecosystem}] CounterfactualDataset: {self.N} patches\n"
            f"  Input window:  t={self.input_start}..{self.input_end-1}\n"
            f"  Forecast:      t={self.post_start}..{self.post_end-1}"
        )
 
    def __len__(self) -> int:
        return self.N
 
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inp    = self.patches[idx, self.input_start:self.input_end].copy()
        actual = self.patches[idx, self.post_start:self.post_end].copy()
        return {
            "input":       torch.from_numpy(inp).float(),
            "actual_post": torch.from_numpy(actual).float(),
            "patch_idx":   torch.tensor(idx, dtype=torch.long),
        }
 
 
# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
 
def build_dataloaders(
    config:           Dict[str, Any],
    ecosystem_patches: Dict[str, str],
    val_split:        float = 0.15,
    seed:             int   = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders with augmentation on train split.
 
    Training augmentations are enabled automatically for the train split.
    Validation split never receives augmentation.
    """
    from torch.utils.data import WeightedRandomSampler
 
    t_input      = config["model"]["t_input"]
    t_output     = config["model"]["t_output"]
    batch_size   = config["training"]["batch_size"]
    window_mode  = config["training"].get("window_mode", "all")
    balanced     = config["training"].get("balanced_sampling", True)
    use_augment  = config["training"].get("use_augmentation", True)
 
    eco_flags = config["training"].get("ecosystems_for_training", {})
    active_ecosystems = [
        eco for eco, enabled in eco_flags.items()
        if enabled and eco in ecosystem_patches
    ]
    if not eco_flags:
        active_ecosystems = list(ecosystem_patches.keys())
 
    logger.info(f"Active ecosystems for training: {active_ecosystems}")
    logger.info(f"Window mode: {window_mode} | Balanced sampling: {balanced} | Augment: {use_augment}")
 
    from data.preprocessing.patch_sampler import PatchSampler
    sampler = PatchSampler(config)
 
    datasets_train = []
    datasets_val   = []
    train_sizes    = []
 
    for eco in active_ecosystems:
        patches_file = Path(ecosystem_patches[eco]) / "patches.npy"
        if not patches_file.exists():
            logger.warning(f"Patches file not found: {patches_file}, skipping {eco}.")
            continue
 
        t_event       = config["ecosystems"][eco]["t_event"]
        patches_shape = np.load(str(patches_file), mmap_mode="r").shape
        T             = patches_shape[1]
 
        windows = sampler.get_windows(
            patches_shape[0], T, t_event, t_input, t_output, mode=window_mode
        )
        if not windows:
            logger.warning(f"[{eco}] No windows generated — skipping.")
            continue
 
        # Split at patch level for spatial separability
        all_patches = list(range(patches_shape[0]))
        rng = np.random.default_rng(seed)
        rng.shuffle(all_patches)
        n_val_patches   = max(1, int(len(all_patches) * val_split))
        val_patches_idx = set(all_patches[:n_val_patches])
        trn_patches_idx = set(all_patches[n_val_patches:])
 
        # Build window lists for train and val patch subsets
        def make_ds(patch_subset, augment):
            # We need a dataset that only uses patches in patch_subset.
            # Easiest: create dataset over all patches, then subset by index.
            full = EcoTimeSeriesDataset(
                patches_file=str(patches_file),
                windows=windows,
                t_input=t_input,
                t_output=t_output,
                ecosystem=eco,
                augment=augment,
                config=config if augment else None,
            )
            n_windows = len(windows)
            indices = [
                p * n_windows + w
                for p in patch_subset
                for w in range(n_windows)
            ]
            from torch.utils.data import Subset
            return Subset(full, indices)
 
        train_ds = make_ds(trn_patches_idx, augment=use_augment)
        val_ds   = make_ds(val_patches_idx, augment=False)
 
        datasets_train.append(train_ds)
        datasets_val.append(val_ds)
        train_sizes.append(len(train_ds))
        logger.info(f"[{eco}] Train: {len(train_ds)} | Val: {len(val_ds)} samples")
 
    if not datasets_train:
        raise RuntimeError(
            "No valid datasets found.\n"
            "  Check that at least one ecosystem is enabled in training.ecosystems_for_training\n"
            "  and that patches exist (run 01_build_tensors.py first)."
        )
 
    combined_train = ConcatDataset(datasets_train) if len(datasets_train) > 1 else datasets_train[0]
    combined_val   = ConcatDataset(datasets_val)   if len(datasets_val)   > 1 else datasets_val[0]
 
    use_sampler = balanced and len(datasets_train) > 1
    if use_sampler:
        weights = []
        for ds_size in train_sizes:
            w = 1.0 / ds_size
            weights.extend([w] * ds_size)
        weights_tensor  = torch.tensor(weights, dtype=torch.float)
        weighted_sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(combined_train),
            replacement=True,
        )
        shuffle_train = False
        sampler_train = weighted_sampler
        logger.info(
            "WeightedRandomSampler enabled: "
            + " | ".join(f"{eco}={sz}" for eco, sz in zip(active_ecosystems, train_sizes))
        )
    else:
        shuffle_train = True
        sampler_train = None
 
    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler_train,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
 
    logger.info(
        f"DataLoaders built: "
        f"train={len(combined_train)} samples ({len(train_loader)} batches) | "
        f"val={len(combined_val)} samples ({len(val_loader)} batches)"
    )
    return train_loader, val_loader