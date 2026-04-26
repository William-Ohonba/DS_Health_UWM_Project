"""
dataset.py — GI Tract Segmentation DataLoader

FIXES applied:
  [FIX-1]  Augmentation no longer zeros neighbor slices.
  [FIX-2]  calc_stats runs on CLAHE-processed images.
  [FIX-5]  /tmp/ split CSVs are now named with the n_slices + img_size
           suffix so parallel runs (n_slices=3 vs 5, img_size=320 vs 448)
           no longer overwrite each other's val split mid-epoch.
<<<<<<< HEAD
  [FIX-7]  Augmentation now passes a 2-D binary union mask as `mask` (for
           MaskAwareRandomCrop anchor sampling) and the full (H,W,C) array
           as `multichan` via additional_targets.
  [FIX-22] get_dataloaders() now accepts a `pin_memory` parameter and passes
           it through to both DataLoaders.
  [FIX-25] WeightedRandomSampler positive weight reduced from 5.0 → 2.0.
  [FIX-30] WeightedRandomSampler removed entirely.
           The 2:1 positive oversampling (FIX-25) still caused a systematic
           train/val calibration mismatch: the model trained on ~67% positive
           slices but validated on ~43% (natural distribution, 57% empty GT).
           This was the primary cause of the optimal threshold oscillating
           between 0.35 and 0.50 epoch-to-epoch and empty_pred% swinging
           26%→50%: the model's sigmoid outputs were calibrated for the wrong
           prior at validation time. Loss class weights (computed per pixel
           frequency in train.py) already compensate for class imbalance at
           the gradient level; the sampler was redundant with those weights
           and actively harmful to calibration. Replaced with shuffle=True.
=======
           Previously both runs wrote to the same /tmp/gi_train_split.csv,
           silently corrupting the val set of whichever started second.
  [FIX-7]  Augmentation now passes a 2-D binary union mask as `mask` (for
           MaskAwareRandomCrop anchor sampling) and the full (H,W,C) array
           as `multichan` via additional_targets.  Previously the raw
           (H,W,3) array arrived in `params["mask"]`, making argwhere
           return (row, col, channel) triples and silently disabling
           mask-focused cropping on every training sample.
>>>>>>> 67bb389d7e8ec687515fe68ebf11894c61af46c5
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ---------------------------------------------------------------------------
# RLE decoder
# ---------------------------------------------------------------------------

def rle_decode(mask_rle, shape):
    """Decode run-length encoding to binary mask."""
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)
    s               = mask_rle.split()
    starts, lengths = (np.asarray(x, dtype=int) for x in (s[0::2], s[1::2]))
    starts         -= 1
    ends            = starts + lengths
    img             = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


# ---------------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------------

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to a [0,1] float32 greyscale image.
    clipLimit=2.0 prevents noise amplification; tileGridSize=(8,8) is
    standard for this dataset.
    """
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def calc_stats(loader, save_path: str = "calcStats.json") -> dict:
    """
    Compute global mean and std over CLAHE-processed images.  [FIX-2]
    Delete calcStats.json and re-run if CLAHE was added after first run.
    """
    from tqdm import tqdm
    print("Computing dataset statistics (CLAHE-processed images) …")
    psum = psum_sq = count = 0.0

    for imgs, _ in tqdm(loader, desc="Stats scan", leave=False):
        psum    += imgs.sum().item()
        psum_sq += (imgs ** 2).sum().item()
        count   += imgs.numel()

    mean  = psum / count
    std   = ((psum_sq / count) - mean ** 2) ** 0.5
    stats = {"mean": mean, "std": std}
    with open(save_path, "w") as f:
        json.dump(stats, f)
    print(f"Stats saved → mean={mean:.4f}, std={std:.4f}")
    return stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GITractDataset(Dataset):
    """
    UW-Madison GI Tract segmentation dataset.
    Supports 2.5D input: N adjacent slices stacked as channels.
    CLAHE applied per slice so each channel is independently normalised.
    """

    CLASSES = ["large_bowel", "small_bowel", "stomach"]

    def __init__(
        self,
        csv_path:   str,
        folder_path: str,
        img_size:   int  = 320,
        stats_path: str  = "calcStats.json",
        augment          = None,
        n_slices:   int  = 3,
        mode:       str  = "train",
    ):
        self.df          = pd.read_csv(csv_path)
        self.folder_path = Path(folder_path)
        self.img_size    = img_size
        self.augment     = augment
        self.n_slices    = n_slices
        self.mode        = mode

        if stats_path and os.path.exists(stats_path):
            with open(stats_path) as f:
                s = json.load(f)
            self.mean = s["mean"]
            self.std  = s["std"]
        else:
            self.mean = self.std = None

<<<<<<< HEAD
        self.ids = self.df["id"].unique()
=======
        self.ids     = self.df["id"].unique()
        self.weights = self._compute_weights()

    def _compute_weights(self):
        """Upweight slices that contain at least one organ (5:1 ratio)."""
        mask_present = (
            self.df.groupby("id")["segmentation"]
            .apply(lambda x: x.notna().any())
        )
        return [5.0 if mask_present.get(sid, False) else 1.0
                for sid in self.ids]
>>>>>>> 67bb389d7e8ec687515fe68ebf11894c61af46c5

    def _parse_id(self, sample_id: str):
        parts    = sample_id.split("_")
        case_id  = parts[0]
        day_id   = parts[1]
        slice_id = "_".join(parts[2:])
        return case_id, day_id, slice_id

    def _load_single_slice(self, folder: Path, slice_num: int) -> np.ndarray:
        """Load one PNG → [0,1] → CLAHE → resize. Returns zeros if missing."""
        matches = list(folder.glob(f"slice_{slice_num:04d}_*.png"))
        if not matches:
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)
        raw = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if raw.max() > 0:
            raw /= raw.max()
        raw = apply_clahe(raw)
        return cv2.resize(raw, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_LINEAR)

    def _load_slice_stack(self, case_id: str, day_id: str, slice_id: str):
        """Load N adjacent slices and stack as channels."""
        slice_num = int(slice_id.split("_")[1])
        half      = self.n_slices // 2
        folder    = self.folder_path / case_id / f"{case_id}_{day_id}" / "scans"
        return np.stack([
            self._load_single_slice(folder, slice_num + off)
            for off in range(-half, half + 1)
        ], axis=0)  # (n_slices, H, W)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id                  = self.ids[idx]
        case_id, day_id, slice_id = self._parse_id(sample_id)

        # ── Image ─────────────────────────────────────────────────────
        if self.n_slices > 1:
            img = self._load_slice_stack(case_id, day_id, slice_id)
        else:
            folder = self.folder_path / case_id / f"{case_id}_{day_id}" / "scans"
            img    = self._load_single_slice(
                folder, int(slice_id.split("_")[1])
            )[np.newaxis]

        # ── Masks ─────────────────────────────────────────────────────
        rows  = self.df[self.df["id"] == sample_id]
        masks = []
        for cls in self.CLASSES:
            row = rows[rows["class"] == cls]
            if len(row) > 0:
                rle    = row["segmentation"].values[0]
                folder = self.folder_path / case_id / f"{case_id}_{day_id}" / "scans"
                matches = list(folder.glob(
                    f"slice_{slice_id.split('_')[1].zfill(4)}_*.png"
                ))
                oh, ow = (int(p) for p in matches[0].stem.split("_")[2:4]) \
                    if matches else (266, 266)
                mask_c = rle_decode(rle, (oh, ow))
                mask_c = cv2.resize(mask_c, (self.img_size, self.img_size),
                                    interpolation=cv2.INTER_NEAREST)
            else:
                mask_c = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            masks.append(mask_c)

        mask = np.stack(masks, axis=0).astype(np.float32)  # (3, H, W)

        # ── Augmentation ──────────────────────────────────────────────
        if self.augment is not None:
            mid_idx = self.n_slices // 2
            mid     = img[mid_idx]                       # (H, W)

            # [FIX-7] build a 2-D union mask for anchor sampling;
            # pass full multi-channel mask via additional_targets so
            # every spatial transform is applied identically to both.
            union_mask   = (mask.max(axis=0) > 0).astype(np.uint8)  # (H, W)
            multichan_hw = mask.transpose(1, 2, 0)                   # (H, W, 3)

            result   = self.augment(
                image     = mid,
                mask      = union_mask,      # 2-D for anchor sampling
                multichan = multichan_hw,    # full (H,W,3) gets same spatial tx
            )
<<<<<<< HEAD
            aug_mid  = result["image"]                         # (new_H, new_W)
=======
            aug_mid  = result["image"]                       # (new_H, new_W)
>>>>>>> 67bb389d7e8ec687515fe68ebf11894c61af46c5
            mask     = result["multichan"].transpose(2, 0, 1)  # (3, new_H, new_W)

            new_h, new_w = aug_mid.shape
            new_img      = np.zeros((self.n_slices, new_h, new_w), dtype=img.dtype)
            for i in range(self.n_slices):
                if i == mid_idx:
                    new_img[i] = aug_mid
                else:
                    # Spatial resize only — neighbor intensity unchanged  [FIX-1]
                    new_img[i] = cv2.resize(img[i], (new_w, new_h),
                                            interpolation=cv2.INTER_LINEAR)
            img = new_img

        # ── Normalise ─────────────────────────────────────────────────
        if self.mean is not None:
            img = (img - self.mean) / (self.std + 1e-6)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    csv_path:      str,
    folder_path:   str,
<<<<<<< HEAD
    img_size:      int   = 320,
    batch_size:    int   = 16,
    n_slices:      int   = 3,
    stats_path:    str   = "calcStats.json",
    train_augment         = None,
    val_split:     float = 0.2,
    num_workers:   int   = 4,
    pin_memory:    bool  = True,   # [FIX-22] threaded from cfg["pin_memory"]
):
    """
    Build train and validation DataLoaders.

    [FIX-22] pin_memory is now a parameter (default True for backward compat)
    rather than being hardcoded. train.py passes cfg["pin_memory"] which is
    True only on CUDA.
    [FIX-30] WeightedRandomSampler removed. Previously 2:1 positive
    oversampling (FIX-25) caused a systematic train/val calibration mismatch.
    Replaced with shuffle=True. Loss class weights in train.py compensate
    for pixel-level class imbalance without distorting the slice distribution.
    """
=======
    img_size:      int  = 320,
    batch_size:    int  = 16,
    n_slices:      int  = 3,
    stats_path:    str  = "calcStats.json",
    train_augment        = None,
    val_split:     float = 0.2,
    num_workers:   int  = 4,
):
    """Build train and validation DataLoaders with weighted sampling."""
>>>>>>> 67bb389d7e8ec687515fe68ebf11894c61af46c5
    df      = pd.read_csv(csv_path)
    all_ids = df["id"].unique()
    n_val   = int(len(all_ids) * val_split)
    np.random.seed(42)
    perm      = np.random.permutation(len(all_ids))
    val_ids   = set(all_ids[perm[:n_val]])
    train_ids = set(all_ids[perm[n_val:]])

    train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["id"].isin(val_ids)].reset_index(drop=True)

    # [FIX-5] Include n_slices + img_size in filename to avoid collisions
    # when running multiple experiments in parallel on the same machine.
    suffix     = f"{n_slices}slice_{img_size}"
    train_csv  = f"/tmp/gi_train_split_{suffix}.csv"
    val_csv    = f"/tmp/gi_val_split_{suffix}.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,     index=False)

    train_ds = GITractDataset(train_csv, folder_path, img_size, stats_path,
                              train_augment, n_slices, "train")
    val_ds   = GITractDataset(val_csv,   folder_path, img_size, stats_path,
                              None,        n_slices, "val")

<<<<<<< HEAD
    # [FIX-30] shuffle=True replaces WeightedRandomSampler.
    # The sampler was causing a train/val distribution mismatch that made
    # the model's optimal threshold oscillate; see module docstring.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,   # [FIX-22]
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,   # [FIX-22]
=======
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(train_ds.weights, dtype=torch.float64),
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
>>>>>>> 67bb389d7e8ec687515fe68ebf11894c61af46c5
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader