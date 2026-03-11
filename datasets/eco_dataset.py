"""
eco_dataset.py
--------------
PyTorch Dataset and DataLoader factory for ECO-REWIND.
 
Each sample is:
  input  : (T_in, C, H, W)  — T_in consecutive normalized quarters
  target : (T_out, C, H, W) — the T_out quarters immediately following
 
Windows are drawn from within the pre-disturbance period only.
Both ecosystems can be combined into a single joint DataLoader.
"""
 
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
 
logger = logging.getLogger(__name__)
 
 
class EcoTimeSeriesDataset(Dataset):
    """
    Sliding-window dataset over pre-disturbance patches.
 
    Parameters
    ----------
    patches_file : path to (N, T, C, H, W) memory-mapped .npy
    windows      : list of (t_start, t_end) tuples — pre-disturbance only
    t_input      : number of input timesteps
    t_output     : number of target timesteps
    ecosystem    : ecosystem name (for logging)
    """
 
    def __init__(
        self,
        patches_file: str,
        windows: List[Tuple[int, int]],
        t_input: int,
        t_output: int,
        ecosystem: str = "unknown",
    ):
        self.ecosystem = ecosystem
        self.t_input = t_input
        self.t_output = t_output
 
        # Load patches as memory-mapped array (no RAM copy)
        self.patches = np.load(patches_file, mmap_mode="r")
        self.N, self.T, self.C, self.H, self.W = self.patches.shape
 
        # Filter windows to those that fit within the tensor
        self.windows = [
            (ts, te) for ts, te in windows
            if te <= self.T and (te - ts) == (t_input + t_output)
        ]
 
        logger.info(
            f"[{ecosystem}] Dataset: {self.N} patches × {len(self.windows)} windows "
            f"= {self.__len__()} samples | C={self.C} H={self.H} W={self.W}"
        )
 
    def __len__(self) -> int:
        return self.N * len(self.windows)
 
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch_idx = idx // len(self.windows)
        window_idx = idx % len(self.windows)
        t_start, t_end = self.windows[window_idx]
 
        # Slice: (T_in + T_out, C, H, W)
        window_data = self.patches[patch_idx, t_start:t_end].copy()
 
        inp = window_data[:self.t_input]         # (T_in, C, H, W)
        target = window_data[self.t_input:]      # (T_out, C, H, W)
 
        return {
            "input": torch.from_numpy(inp).float(),
            "target": torch.from_numpy(target).float(),
            "patch_idx": torch.tensor(patch_idx, dtype=torch.long),
            "t_start": torch.tensor(t_start, dtype=torch.long),
            "ecosystem": self.ecosystem,
        }
 
 
class CounterfactualDataset(Dataset):
    """
    Dataset for counterfactual inference — feeds pre-disturbance context
    and asks the model to predict forward past the hurricane event.
 
    Unlike training, windows here START in the pre-disturbance window
    and EXTEND past t_event into the post-disturbance period.
 
    Each sample covers the full temporal context needed for one patch.
    """
 
    def __init__(
        self,
        patches_file: str,
        t_event: int,
        t_input: int,
        n_forecast: int,
        ecosystem: str = "unknown",
    ):
        self.ecosystem = ecosystem
        self.t_event = t_event
        self.t_input = t_input
        self.n_forecast = n_forecast
 
        self.patches = np.load(patches_file, mmap_mode="r")
        self.N, self.T, self.C, self.H, self.W = self.patches.shape
 
        # Input window: [t_event - t_input : t_event]
        self.input_start = max(0, t_event - t_input)
        self.input_end = t_event
 
        # Actual post-event data: [t_event : t_event + n_forecast]
        self.post_start = t_event
        self.post_end = min(self.T, t_event + n_forecast)
 
        logger.info(
            f"[{ecosystem}] CounterfactualDataset: {self.N} patches\n"
            f"  Input window:  t={self.input_start}..{self.input_end-1}\n"
            f"  Forecast:      t={self.post_start}..{self.post_end-1}"
        )
 
    def __len__(self) -> int:
        return self.N
 
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Pre-event context
        inp = self.patches[idx, self.input_start:self.input_end].copy()
 
        # Actual post-event trajectory
        actual = self.patches[idx, self.post_start:self.post_end].copy()
 
        return {
            "input": torch.from_numpy(inp).float(),
            "actual_post": torch.from_numpy(actual).float(),
            "patch_idx": torch.tensor(idx, dtype=torch.long),
        }
 
 
# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
 
def build_dataloaders(
    config: Dict[str, Any],
    ecosystem_patches: Dict[str, str],   # {"everglades": path, "mississippi": path}
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.
 
    Reads three training config keys:
      ecosystems_for_training: {everglades: true, mississippi: false}
        Controls which ecosystems are included. Flip mississippi to true
        when v2 (Barataria Bay) re-export is available.
 
      window_mode: "all" | "pre_only"
        "all"      → windows slide across full 32-quarter time series (~28/patch)
        "pre_only" → windows restricted to pre-disturbance period only (old behaviour)
 
      balanced_sampling: true | false
        When multiple ecosystems are active, use WeightedRandomSampler to give
        each ecosystem equal representation in every batch.
 
    Val split is done at the patch level (spatially separate patches).
    Returns: (train_loader, val_loader)
    """
    from torch.utils.data import WeightedRandomSampler
 
    t_input = config["model"]["t_input"]
    t_output = config["model"]["t_output"]
    batch_size = config["training"]["batch_size"]
    window_mode = config["training"].get("window_mode", "all")
    balanced = config["training"].get("balanced_sampling", True)
 
    # Determine which ecosystems are active for training
    eco_flags = config["training"].get("ecosystems_for_training", {})
    active_ecosystems = [
        eco for eco, enabled in eco_flags.items()
        if enabled and eco in ecosystem_patches
    ]
    # Fallback: if flag dict not present, use all available patches
    if not eco_flags:
        active_ecosystems = list(ecosystem_patches.keys())
 
    logger.info(f"Active ecosystems for training: {active_ecosystems}")
    logger.info(f"Window mode: {window_mode} | Balanced sampling: {balanced}")
 
    from data.preprocessing.patch_sampler import PatchSampler
    sampler = PatchSampler(config)
 
    datasets_train = []
    datasets_val = []
    train_sizes = []   # used for balanced sampler weights
 
    for eco in active_ecosystems:
        patches_file = Path(ecosystem_patches[eco]) / "patches.npy"
        if not patches_file.exists():
            logger.warning(f"Patches file not found: {patches_file}, skipping {eco}.")
            continue
 
        t_event = config["ecosystems"][eco]["t_event"]
        patches_shape = np.load(str(patches_file), mmap_mode="r").shape
        T = patches_shape[1]
 
        windows = sampler.get_windows(
            patches_shape[0], T, t_event, t_input, t_output, mode=window_mode
        )
 
        if not windows:
            logger.warning(f"[{eco}] No windows generated — skipping.")
            continue
 
        full_dataset = EcoTimeSeriesDataset(
            patches_file=str(patches_file),
            windows=windows,
            t_input=t_input,
            t_output=t_output,
            ecosystem=eco,
        )
 
        n_val = max(1, int(len(full_dataset) * val_split))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )
        datasets_train.append(train_ds)
        datasets_val.append(val_ds)
        train_sizes.append(n_train)
        logger.info(f"[{eco}] Train: {n_train} | Val: {n_val} samples")
 
    if not datasets_train:
        raise RuntimeError(
            "No valid datasets found.\n"
            "  Check that at least one ecosystem is enabled in training.ecosystems_for_training\n"
            "  and that patches exist (run 01_build_tensors.py first)."
        )
 
    combined_train = ConcatDataset(datasets_train) if len(datasets_train) > 1 else datasets_train[0]
    combined_val = ConcatDataset(datasets_val) if len(datasets_val) > 1 else datasets_val[0]
 
    # Build WeightedRandomSampler for balanced multi-ecosystem training.
    # Each sample gets weight = 1 / (ecosystem_size), so each ecosystem
    # contributes equally to every batch regardless of its sample count.
    use_sampler = balanced and len(datasets_train) > 1
    if use_sampler:
        weights = []
        for ds_size in train_sizes:
            w = 1.0 / ds_size
            weights.extend([w] * ds_size)
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        weighted_sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(combined_train),
            replacement=True,
        )
        shuffle_train = False
        sampler_train = weighted_sampler
        logger.info(
            f"WeightedRandomSampler enabled: "
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