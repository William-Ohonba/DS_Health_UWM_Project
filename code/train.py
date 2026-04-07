"""
train.py — Full training loop for GI Tract Segmentation
Implements everything from Presentation 3:
  - AdamW optimizer + CosineAnnealingLR scheduler
  - Combined BCE + Dice loss
  - WeightedRandomSampler for empty-slice imbalance
  - Dice + Hausdorff metric tracking
  - Best model checkpointing

Presentation 3 experiment design:
  Run with --n_slices 3  →  2.5D model using 3 adjacent slices
  Run with --n_slices 5  →  2.5D model using 5 adjacent slices
  Results are saved to separate checkpoints and logs for direct comparison.
"""

import os # for checking if a certain file path exists
import json # for saving the training log history
import argparse # handles command line arguements (--n_slices and --fast)
import numpy as np # gives basic computation methods like .mean
import torch # gives access to tensors and tensor manipulation
import torch.optim as optim # gives access to Adam optimization
from torch.optim.lr_scheduler import CosineAnnealingLR # use cosine curve formula to make a learning rate scheduler
from tqdm import tqdm # allows progress bar
from pathlib import Path

from dataset       import get_dataloaders, calc_stats, GITractDataset # dataloader wraps dataset and allows handling of dataset, calc_stat allows statistic calculation for dataset(mean, SD), GITrackDataset is the dataset
from model         import build_model #constructs the UNET model
from loss          import CombinedLoss, compute_metrics # Combined loss gives the loss function(BCE and DICE), compute_metrics computes DICE and Hausdorff
from augmentations import get_train_augmentations # gets the albumentation pipeline, sequence of image transformations in computer vision tasks


# ---------------------------------------------------------------------------
# Base configuration — paths and hyperparameters shared by all runs
# ---------------------------------------------------------------------------
_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
) # if no gpu use cpu, if using nvidia gpu use cuda, if using mac use mps using torch cuda and torch backends

DATA_DIR = Path("data")

BASE_CONFIG = {
    "csv_path"    : str(DATA_DIR / "train.csv"), # set appropriate path depending on where your train.csv is
    "folder_path" : str(DATA_DIR / "train"), # set appropriate path for where you train folder is
    "stats_path"  : "calcStats.json", #where will your statistics go, in this case calcStats.jjson
    # img_size 224 → ~6× faster than 320; good enough to compare 3- vs 5-slice
    # bump to 320 for the final presentation run
    "img_size"    : 224,
    "batch_size"  : 16,         # 224² fits more per batch; helps MPS throughput
    "num_workers" : 2,          # MPS: more workers → more IPC overhead, keep low
    "pin_memory"  : _DEVICE == "cuda",   # MPS doesn't support pin_memory
    "epochs"      : 100,         # enough to see 3- vs 5-slice differences; ~8 hrs total
    "lr"          : 1e-4,       # learning rate passed into Adam
    "device"      : _DEVICE,    #assigns device to the earlier  assigned device
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="GI Tract Segmentation — 2.5D comparison experiment"
    )
    #provides the --n_slices arguement
    parser.add_argument(
        "--n_slices",
        type=int,
        choices=[3, 5],
        default=3,
        help="Number of adjacent slices for 2.5D input (3 or 5). "
             "Run once with each value to generate Presentation 3 comparison results.",
    )
    # provides the ability to test the train file on a smaller faster implementation
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke-test mode: 3 epochs, 200 train batches, 50 val batches. "
             "Use to confirm everything runs before committing to a full experiment.",
    )
    return parser.parse_args()

# combines the base configs with the configs of if you chose 3 slices or 5 slices
def build_run_config(n_slices: int) -> dict:
    """Merge base config with run-specific settings."""
    cfg = BASE_CONFIG.copy()
    cfg["n_slices"]   = n_slices
    cfg["checkpoint"] = f"best_model_{n_slices}slice.pth"
    cfg["log_file"]   = f"training_log_{n_slices}slice.json"
    return cfg


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train() # sets the given model to training mode
    total_loss = 0.0 # sum variable for eventual average loss

    for imgs, masks in tqdm(loader, desc="Train", leave=False): #iterates through the batchs of dataloader wrapped by tqdm for progress bar
        imgs  = imgs.to(device) # moves tensor to appropriate device
        masks = masks.to(device) # moves tensor to appropriate device

        optimizer.zero_grad() # zero's out the gradiant to clear out any leftover

        if scaler is not None:          # Mixed precision (CUDA only)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad() # turns off tracking because back propogation isnt used to save resources
def validate(model, loader, criterion, device):
    model.eval() # puts model in evaluate mode to prevent neuron dropping
    total_loss  = 0.0 # collects loss scalar
    all_metrics = [] # collects metrics in array

    for imgs, masks in tqdm(loader, desc="Val  ", leave=False): # loops data loader with progress bar
        imgs  = imgs.to(device) # moves image tensor to appropriate gpu/cpu (mps) 
        masks = masks.to(device) # moves mask tensor to appropriate gpu/cpu (mps)

        logits = model(imgs) # forward pass to get unnormalized predictions
        loss   = criterion(logits, masks) # compares 
        total_loss += loss.item()

        metrics = compute_metrics(logits, masks)
        all_metrics.append(metrics)

    avg_loss = total_loss / len(loader)
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0]
    }
    return avg_loss, avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = build_run_config(args.n_slices)

    # --fast: override to a quick smoke-test so you can verify the full pipeline
    # in ~5 minutes before committing to an overnight run
    if args.fast:
        cfg["epochs"]     = 3
        cfg["checkpoint"] = f"fast_model_{args.n_slices}slice.pth"
        cfg["log_file"]   = f"fast_log_{args.n_slices}slice.json"

    device = cfg["device"]
    print(f"\n{'='*60}")
    print(f"  2.5D GI Tract Segmentation — {cfg['n_slices']}-slice experiment")
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {cfg['checkpoint']}")
    print(f"  Log file  : {cfg['log_file']}")
    print(f"{'='*60}\n")

    # ---- Dataset stats (computed once, shared across both experiments) ----
    if not os.path.exists(cfg["stats_path"]):
        print("calcStats.json not found — computing dataset stats...")
        tmp_ds = GITractDataset(
            cfg["csv_path"], cfg["folder_path"],
            img_size=cfg["img_size"], stats_path=None,
        )
        from torch.utils.data import DataLoader
        tmp_loader = DataLoader(tmp_ds, batch_size=16, num_workers=cfg["num_workers"])
        calc_stats(tmp_loader, cfg["stats_path"])
        print("Stats saved to calcStats.json — reused by the other experiment.\n")

    # ---- DataLoaders ----
    train_aug = get_train_augmentations(cfg["img_size"])
    train_loader, val_loader = get_dataloaders(
        csv_path      = cfg["csv_path"],
        folder_path   = cfg["folder_path"],
        img_size      = cfg["img_size"],
        batch_size    = cfg["batch_size"],
        n_slices      = cfg["n_slices"],
        stats_path    = cfg["stats_path"],
        train_augment = train_aug,
        num_workers   = cfg["num_workers"],
    )

    # --fast: cap loader lengths so the smoke-test finishes quickly
    if args.fast:
        from itertools import islice
        _train = list(islice(train_loader, 200))
        _val   = list(islice(val_loader, 50))
        train_loader = _train
        val_loader   = _val
        print("  [fast mode] capped to 200 train / 50 val batches\n")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = build_model(
        n_slices   = cfg["n_slices"],
        n_classes  = 3,
        pretrained = True,
        device     = device,
    )

    # ---- Loss, Optimizer, Scheduler ----
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    # Mixed-precision scaler (GPU only)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # ---- Training Loop ----
    best_dice   = 0.0
    log_history = []

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}  lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss            = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        dice_mean = val_metrics["dice_mean"]
        hd_mean   = val_metrics["hausdorff_mean"]
        composite = val_metrics["composite"]

        print(
            f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}\n"
            f"  Dice  — large_bowel: {val_metrics['dice_large_bowel']:.3f}  "
            f"small_bowel: {val_metrics['dice_small_bowel']:.3f}  "
            f"stomach: {val_metrics['dice_stomach']:.3f}  mean: {dice_mean:.3f}\n"
            f"  HD    — mean: {hd_mean:.4f}\n"
            f"  Composite Score: {composite:.4f}"
        )

        # Save best model
        if dice_mean > best_dice:
            best_dice = dice_mean
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "dice_mean"  : dice_mean,
                "config"     : cfg,
            }, cfg["checkpoint"])
            print(f"  ✓ Saved best model (Dice={best_dice:.4f})")

        # Append to log
        log_history.append({
            "epoch"      : epoch,
            "train_loss" : train_loss,
            "val_loss"   : val_loss,
            **{k: float(v) for k, v in val_metrics.items()},
        })
        with open(cfg["log_file"], "w") as f:
            json.dump(log_history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"  Best Dice : {best_dice:.4f}")
    print(f"  Checkpoint: {cfg['checkpoint']}")
    print(f"  Log       : {cfg['log_file']}")


if __name__ == "__main__":
    main()