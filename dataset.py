"""
dataset.py — GI Tract Segmentation DataLoader
Handles: RLE decoding, CLAHE, normalization, 2.5D slice stacking
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path


def rle_decode(mask_rle, shape):
    """Decode run-length encoding to binary mask."""
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def apply_clahe(img):
    """Apply CLAHE to normalize MRI scanner intensity drift."""
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_uint8)
    return img_clahe.astype(np.float32) / 255.0


def calc_stats(loader, save_path="calcStats.json"):
    """Compute global mean and std across the entire dataset."""
    print("Computing dataset statistics...")
    psum, psum_sq, count = 0.0, 0.0, 0

    for imgs, _ in loader:
        psum    += imgs.sum().item()
        psum_sq += (imgs ** 2).sum().item()
        count   += imgs.numel()

    mean = psum / count
    std  = ((psum_sq / count) - (mean ** 2)) ** 0.5

    stats = {"mean": mean, "std": std}
    with open(save_path, "w") as f:
        json.dump(stats, f)
    print(f"Stats saved → mean={mean:.4f}, std={std:.4f}")
    return stats


class GITractDataset(Dataset):
    """
    Dataset for UW-Madison GI Tract segmentation.
    Supports 2.5D: stacks N adjacent slices as input channels.

    Args:
        csv_path    : path to train.csv
        folder_path : path to the /train root folder
        img_size    : target image size (square)
        stats_path  : path to calcStats.json (None = no normalization)
        augment     : albumentations transform (or None)
        n_slices    : number of adjacent slices to stack (1 = standard 2D, 3 = 2.5D)
        mode        : 'train' | 'val' | 'test'
    """

    CLASSES = ["large_bowel", "small_bowel", "stomach"]

    def __init__(
        self,
        csv_path,
        folder_path,
        img_size=320,
        stats_path="calcStats.json",
        augment=None,
        n_slices=3,
        mode="train",
    ):
        self.df          = pd.read_csv(csv_path)
        self.folder_path = Path(folder_path)
        self.img_size    = img_size
        self.augment     = augment
        self.n_slices    = n_slices
        self.mode        = mode

        # Load normalization stats
        if stats_path and os.path.exists(stats_path):
            with open(stats_path) as f:
                s = json.load(f)
            self.mean = s["mean"]
            self.std  = s["std"]
        else:
            self.mean = None
            self.std  = None

        # Group by unique slice (id = case_day_slice)
        self.ids = self.df["id"].unique()

        # Precompute sample weights for WeightedRandomSampler
        self.weights = self._compute_weights()

    def _compute_weights(self):
        """Upweight slices that contain at least one organ."""
        mask_present = (
            self.df.groupby("id")["segmentation"]
            .apply(lambda x: x.notna().any())
        )
        weights = []
        for sid in self.ids:
            weights.append(5.0 if mask_present.get(sid, False) else 1.0)
        return weights

    def _parse_id(self, sample_id):
        parts    = sample_id.split("_")
        case_id  = parts[0]
        day_id   = parts[1]
        slice_id = "_".join(parts[2:])
        return case_id, day_id, slice_id

    def _load_image(self, case_id, day_id, slice_id):
        folder    = self.folder_path / case_id / f"{case_id}_{day_id}" / "scans"
        matches   = list(folder.glob(f"slice_{slice_id}_*.png"))
        if not matches:
            return None
        img_path  = matches[0]
        img       = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Scale to [0, 1]
        if img.max() > 0:
            img = img / img.max()

        # CLAHE
        img = apply_clahe(img)

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return img

    def _load_slice_stack(self, case_id, day_id, slice_id):
        """Load N adjacent slices and stack as channels for 2.5D input."""
        slice_num = int(slice_id.split("_")[1])
        half      = self.n_slices // 2
        stack     = []

        for offset in range(-half, half + 1):
            neighbor_id = f"slice_{slice_num + offset:04d}_"
            folder      = self.folder_path / case_id / f"{case_id}_{day_id}" / "scans"
            matches     = list(folder.glob(f"slice_{slice_num + offset:04d}_*.png"))

            if matches:
                img = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED).astype(np.float32)
                if img.max() > 0:
                    img = img / img.max()
                img = apply_clahe(img)
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            else:
                # Pad with zeros if neighbor doesn't exist
                img = np.zeros((self.img_size, self.img_size), dtype=np.float32)

            stack.append(img)

        return np.stack(stack, axis=0)  # (n_slices, H, W)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id          = self.ids[idx]
        case_id, day_id, slice_id = self._parse_id(sample_id)

        # Load image stack (2.5D)
        if self.n_slices > 1:
            img = self._load_slice_stack(case_id, day_id, slice_id)  # (n_slices, H, W)
        else:
            raw = self._load_image(case_id, day_id, slice_id)
            if raw is None:
                raw = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            img = raw[np.newaxis, ...]  # (1, H, W)

        # Load masks for all 3 classes
        rows   = self.df[self.df["id"] == sample_id]
        h, w   = self.img_size, self.img_size
        masks  = []
        for cls in self.CLASSES:
            row = rows[rows["class"] == cls]
            if len(row) > 0:
                rle  = row["segmentation"].values[0]
                # Get original image dimensions from filename
                folder = self.folder_path / case_id / f"{case_id}_{day_id}" / "scans"
                matches = list(folder.glob(f"slice_{slice_id}_*.png"))
                if matches:
                    fname  = matches[0].name
                    parts  = fname.replace(".png", "").split("_")
                    oh, ow = int(parts[-2]), int(parts[-1])
                else:
                    oh, ow = 266, 266
                mask = rle_decode(rle, (oh, ow))
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            masks.append(mask)

        mask = np.stack(masks, axis=0).astype(np.float32)  # (3, H, W)

        # Augmentation (applied to middle slice + mask)
        if self.augment is not None:
            mid    = img[self.n_slices // 2]  # use center slice for aug
            result = self.augment(image=mid, mask=mask.transpose(1, 2, 0))
            mid    = result["image"]
            mask   = result["mask"].transpose(2, 0, 1)
            img[self.n_slices // 2] = mid

        # Normalize
        if self.mean is not None:
            img = (img - self.mean) / (self.std + 1e-6)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


def get_dataloaders(
    csv_path,
    folder_path,
    img_size=320,
    batch_size=16,
    n_slices=3,
    stats_path="calcStats.json",
    train_augment=None,
    val_split=0.2,
    num_workers=4,
):
    """Build train and validation DataLoaders with weighted sampling."""
    df         = pd.read_csv(csv_path)
    all_ids    = df["id"].unique()
    n_val      = int(len(all_ids) * val_split)
    np.random.seed(42)
    perm       = np.random.permutation(len(all_ids))
    val_ids    = set(all_ids[perm[:n_val]])
    train_ids  = set(all_ids[perm[n_val:]])

    train_df   = df[df["id"].isin(train_ids)].reset_index(drop=True)
    val_df     = df[df["id"].isin(val_ids)].reset_index(drop=True)

    train_csv  = "/tmp/gi_train_split.csv"
    val_csv    = "/tmp/gi_val_split.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,   index=False)

    train_ds = GITractDataset(train_csv, folder_path, img_size, stats_path, train_augment, n_slices, "train")
    val_ds   = GITractDataset(val_csv,   folder_path, img_size, stats_path, None,          n_slices, "val")

    # Weighted sampler to address empty-slice imbalance
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(train_ds.weights),
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
