"""
train.py — Full training loop for GI Tract Segmentation

Fixes applied (cumulative):
  [FIX-1]  dataset.py: augmentation no longer zeros neighbor slices
  [FIX-2]  calc_stats runs on CLAHE-processed images
  [FIX-3]  Dice loss smooth: 1.0 → 1e-6  (restores gradient signal)
  [FIX-4]  Differential LR: encoder=3e-5, decoder/head=3e-4
  [FIX-5]  calcStats.json stale-stats warning on startup
           /tmp/ split CSVs now namespaced by n_slices+img_size (dataset.py)
  [FIX-6]  Dice threshold: 0.5 → 0.3 in compute_dice_safe during validation
  [FIX-7]  Hausdorff Distance actually computed in validate()
           augmentations.py / dataset.py: union mask passed for crop anchor
  [FIX-8]  Small bowel class weight boosted ×1.5 after frequency calculation
  [FIX-9]  Sampler generator reassignment removed — it was a no-op because
           the DataLoader captures the generator reference at construction
           time; reassigning the attribute on the sampler mid-run did nothing.
           Reproducible ordering is now achieved by seeding worker_init_fn.
  [FIX-10] GradScaler state saved + restored in crash checkpoint so AMP
           scale factor is correct after --resume, not reset to default.
  [FIX-11] hausdorff_distance_2d returns None for both-empty pairs (loss.py);
           validate() skips those pairs from the HD mean — matches the
           skip-both-empty logic in compute_dice_safe().
  [FIX-12] Dice and HD now both evaluated at the SAME threshold (0.3) so
           the composite score 0.4·Dice + 0.6·(1−HD) is coherent.
           Previously Dice used 0.3 and HD used 0.5 in the same call.
  [FIX-13] HD computed only every HD_EVAL_EVERY epochs (default 5) and on
           a random subsample of HD_EVAL_BATCHES batches (default 40).
           Previously it ran on all ~23 000 val pairs every epoch, making
           each validation pass ~4× slower than necessary.
  [FIX-14] save_mask_overlay wraps body in try/finally so model.train() is
           always called even if an exception occurs mid-visualisation.
  [FIX-15] Threshold sweep: after every epoch, compute_dice_safe is evaluated
           at thresholds [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] on the
           validation logit cache and the best threshold is stored. This
           directly addresses the Dice oscillation seen at epochs 30-37,
           which was caused by the model's output distribution shifting while
           the threshold was fixed, making each epoch's Dice score dependent
           on how well 0.3 happened to match that epoch's calibration.

Run:
    python train.py --n_slices 3
    python train.py --n_slices 5
    python train.py --n_slices 3 --fast
    python train.py --n_slices 3 --resume
    python train.py --n_slices 3 --upscale

TensorBoard:
    tensorboard --logdir runs/
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import random
import argparse
import multiprocessing
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (CosineAnnealingLR, LinearLR,
                                       SequentialLR, ReduceLROnPlateau)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from dataset       import get_dataloaders, calc_stats, GITractDataset
from model         import build_model
from loss          import hausdorff_distance_2d
from augmentations import get_train_augmentations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES  = ["large_bowel", "small_bowel", "stomach"]
CLASS_COLORS = [(255, 80, 80), (80, 200, 80), (80, 120, 255)]

IDX_LARGE_BOWEL = 0
IDX_SMALL_BOWEL = 1
IDX_STOMACH     = 2

# HD is expensive — evaluate every N epochs on M batches only  [FIX-13]
HD_EVAL_EVERY  = 5
HD_EVAL_BATCHES = 40

# Thresholds swept each epoch to find the best operating point  [FIX-15]
THRESHOLD_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


# ---------------------------------------------------------------------------
# Optimal worker count
# ---------------------------------------------------------------------------

def optimal_workers(device: str) -> int:
    cpus = multiprocessing.cpu_count()
    if device == "cuda":
        return min(cpus, 8)
    elif device == "mps":
        return min(cpus, 4)
    return max(1, cpus // 2)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

import os as _os
_current_dir = _os.path.dirname(_os.path.abspath(__file__))
from pathlib import Path as _Path
DATA_DIR = _Path(_current_dir) / "uw-madison-gi-tract-image-segmentation"

BASE_CONFIG = {
    "csv_path"     : str(DATA_DIR / "train.csv"),
    "folder_path"  : str(DATA_DIR / "train"),
    "stats_path"   : "calcStats.json",
    "img_size"     : 320,
    "batch_size"   : 8,
    "accum_steps"  : 2,        # simulates batch_size=16
    "epochs"       : 100,
    "lr"           : 3e-5,     # encoder LR — decoder gets 10× this
    "warmup_epochs": 5,
    "device"       : _DEVICE,
}


# ---------------------------------------------------------------------------
# Per-class frequency weighting
# ---------------------------------------------------------------------------

def compute_class_weights(loader, n_classes: int = 3,
                          device: str = "cpu") -> torch.Tensor:
    """
    Scan the training loader once and return inverse-frequency class weights.
    [FIX-8] Small bowel receives an additional ×1.5 boost on top of the
    inverse-frequency weight because it consistently scores ~40% lower.
    """
    print("Computing per-class pixel frequencies for loss weighting …")
    class_pixel_sum = np.zeros(n_classes, dtype=np.float64)
    total_pixels    = 0

    for _, masks in tqdm(loader, desc="Class freq scan", leave=False):
        for c in range(n_classes):
            class_pixel_sum[c] += masks[:, c].sum().item()
        total_pixels += masks[:, 0].numel() * masks.shape[0]

    freq    = class_pixel_sum / (total_pixels + 1e-8)
    weights = 1.0 / (freq + 1e-4)
    weights[IDX_SMALL_BOWEL] *= 1.5                    # [FIX-8]
    weights = weights / weights.sum() * n_classes      # renormalise → mean=1

    print(f"  Class frequencies : {freq}")
    print(f"  Class loss weights: {weights}  (small_bowel boosted ×1.5)")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Weighted per-class loss
# ---------------------------------------------------------------------------

class WeightedPerClassCombinedLoss(torch.nn.Module):
    """BCE+Dice computed per class then combined with frequency weights."""

    def __init__(self, class_weights: torch.Tensor,
                 bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.bce   = torch.nn.BCEWithLogitsLoss(reduction="none")

    def _dice_loss(self, logits, targets, smooth: float = 1e-6):
        """[FIX-3] smooth=1e-6 so near-zero preds yield loss≈1.0."""
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice  = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
        return dice.mean(dim=0)                        # [C]

    def forward(self, logits, targets):
        bce_per_class  = self.bce(logits, targets).mean(dim=(0, 2, 3))  # [C]
        dice_per_class = self._dice_loss(logits, targets)               # [C]
        loss_per_class = self.bce_w * bce_per_class + self.dice_w * dice_per_class
        weighted_loss  = (loss_per_class * self.class_weights).sum() \
                       / self.class_weights.sum()
        return weighted_loss, loss_per_class.detach()


# ---------------------------------------------------------------------------
# Dice metric with threshold sweep  [FIX-12, FIX-15]
# ---------------------------------------------------------------------------

def compute_dice_at_threshold(logits_list: list,
                               masks_list:  list,
                               threshold:   float,
                               smooth:      float = 1e-6) -> dict:
    """
    Compute per-class Dice over pre-collected (logits, mask) pairs.
    Skips both-empty pairs. Used by the threshold sweep [FIX-15].

    logits_list / masks_list: lists of CPU numpy arrays [B, C, H, W].
    """
    per_class = {c: [] for c in CLASS_NAMES}

    for logits_np, masks_np in zip(logits_list, masks_list):
        probs = 1.0 / (1.0 + np.exp(-logits_np))   # sigmoid
        preds = (probs > threshold).astype(np.float32)

        for c, cname in enumerate(CLASS_NAMES):
            for b in range(logits_np.shape[0]):
                gt_sum   = masks_np[b, c].sum()
                pred_sum = preds[b, c].sum()
                if gt_sum == 0 and pred_sum == 0:
                    continue
                inter = (preds[b, c] * masks_np[b, c]).sum()
                denom = pred_sum + gt_sum
                per_class[cname].append(
                    (2.0 * inter + smooth) / (denom + smooth)
                )

    result      = {}
    valid_means = []
    for cname in CLASS_NAMES:
        vals = per_class[cname]
        v    = float(np.mean(vals)) if vals else float("nan")
        result[f"dice_{cname}"] = v
        if not np.isnan(v):
            valid_means.append(v)

    result["dice_mean"] = float(np.mean(valid_means)) if valid_means else float("nan")
    return result


def sweep_threshold(logits_list: list, masks_list: list) -> tuple:
    """
    Evaluate Dice at each threshold in THRESHOLD_SWEEP.
    Returns (best_threshold, best_dice_mean, all_results_dict).
    [FIX-15] Addresses epoch 30-37 oscillation caused by fixed threshold
    not tracking the model's shifting output distribution.
    """
    best_t     = THRESHOLD_SWEEP[0]
    best_dice  = -1.0
    all_results = {}

    for t in THRESHOLD_SWEEP:
        res  = compute_dice_at_threshold(logits_list, masks_list, t)
        dm   = res["dice_mean"]
        all_results[t] = res
        if not np.isnan(dm) and dm > best_dice:
            best_dice = dm
            best_t    = t

    return best_t, best_dice, all_results


# ---------------------------------------------------------------------------
# Mask visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_mask_overlay(model, sample_img, sample_mask,
                      save_path, epoch, device, threshold: float = 0.5):
    """
    Save prediction overlay PNG.
    [FIX-14] model.train() guaranteed via try/finally even on exception.
    """
    model.eval()
    try:
        img_t  = sample_img.unsqueeze(0).to(device)
        logits = model(img_t)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()
        pred   = (probs > threshold).astype(np.uint8)

        mid_idx = sample_img.shape[0] // 2
        base    = sample_img[mid_idx].cpu().numpy()
        base    = np.clip(base, 0, 1)
        p2, p98 = np.percentile(base, 2), np.percentile(base, 98)
        if p98 > p2:
            base = (base - p2) / (p98 - p2)
        base = np.clip(base, 0, 1)

        vis = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        gt  = sample_mask.cpu().numpy().astype(np.uint8)

        for c, color in enumerate(CLASS_COLORS):
            if pred[c].any():
                overlay = vis.copy()
                overlay[pred[c] == 1] = color
                vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
            if gt[c].any():
                contours, _ = cv2.findContours(
                    gt[c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, color, 2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(f"Epoch {epoch} — Segmentation Preview",
                     fontsize=14, fontweight="bold", color="white")

        axes[0].imshow(vis)
        axes[0].set_title("Prediction (filled) + GT (outline)", color="white")
        axes[0].axis("off")

        prob_rgb = np.zeros((*probs.shape[1:], 3), dtype=np.float32)
        for c, color in enumerate(CLASS_COLORS):
            for ch, v in enumerate(color):
                prob_rgb[..., ch] += probs[c] * (v / 255.0)
        prob_rgb = np.clip(prob_rgb, 0, 1)

        axes[1].imshow(base, cmap="gray")
        axes[1].imshow(prob_rgb, alpha=0.6)
        axes[1].set_title("Probability Heatmap", color="white")
        axes[1].axis("off")

        for ax in axes:
            ax.set_facecolor("#0d0d1a")

        patches = [
            mpatches.Patch(color=np.array(c) / 255, label=CLASS_NAMES[i])
            for i, c in enumerate(CLASS_COLORS)
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3,
                   facecolor="#1a1a2e", labelcolor="white", fontsize=10)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    finally:
        model.train()   # [FIX-14] always restore training mode


# ---------------------------------------------------------------------------
# Argument parsing / config
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_slices", type=int, choices=[3, 5], default=3)
    parser.add_argument("--upscale",  action="store_true")
    parser.add_argument("--fast",     action="store_true")
    parser.add_argument("--resume",   action="store_true")
    return parser.parse_args()


def build_run_config(n_slices: int, fast: bool, upscale: bool) -> dict:
    cfg                  = BASE_CONFIG.copy()
    cfg["n_slices"]      = n_slices
    cfg["img_size"]      = 448 if upscale else 320
    cfg["num_workers"]   = optimal_workers(_DEVICE)
    cfg["pin_memory"]    = _DEVICE == "cuda"
    suffix               = "_448" if upscale else "_320"
    cfg["checkpoint"]         = f"best_model_{n_slices}slice{suffix}.pth"
    cfg["crash_checkpoint"]   = f"crash_recovery_{n_slices}slice.pth"
    cfg["log_file"]           = f"training_log_{n_slices}slice.json"
    cfg["tb_run_name"]        = f"gi_{n_slices}slice"
    cfg["vis_dir"]            = f"vis_{n_slices}slice"
    if fast:
        cfg["epochs"]           = 3
        cfg["checkpoint"]       = f"fast_model_{n_slices}slice.pth"
        cfg["crash_checkpoint"] = f"fast_crash_{n_slices}slice.pth"
        cfg["log_file"]         = f"fast_log_{n_slices}slice.json"
        cfg["tb_run_name"]      = f"gi_fast_{n_slices}slice"
        cfg["vis_dir"]          = f"vis_fast_{n_slices}slice"
    return cfg


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion,
                    device, scaler, writer, global_step,
                    epoch, fast, accum_steps=2, fast_limit=200):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)
    for batch_idx, (imgs, masks) in enumerate(pbar):
        if fast and batch_idx >= fast_limit:
            break

        imgs  = imgs.to(device)
        masks = masks.to(device)

        is_update = ((batch_idx + 1) % accum_steps == 0) or \
                    (batch_idx + 1 == len(loader))
        if batch_idx % accum_steps == 0:
            optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits, per_class = _forward(model, criterion, imgs, masks)
            scaler.scale(logits / accum_steps).backward()
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss, per_class = _forward(model, criterion, imgs, masks)
            (loss / accum_steps).backward()
            if is_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            logits = loss   # reuse variable name for logging below

        loss_val = logits.item() if hasattr(logits, "item") else 0.0
        total_loss += loss_val
        n_batches  += 1
        global_step += 1

        writer.add_scalar("Batch/train_loss", loss_val, global_step)
        for c, cname in enumerate(CLASS_NAMES):
            writer.add_scalar(f"Batch/loss_{cname}", per_class[c].item(), global_step)

        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return total_loss / max(n_batches, 1), global_step


def _forward(model, criterion, imgs, masks):
    logits = model(imgs)
    loss, per_class = criterion(logits, masks)
    return loss, per_class


# ---------------------------------------------------------------------------
# Validation loop  [FIX-12, FIX-13, FIX-15]
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device, fast,
             epoch: int = 1,
             compute_hd: bool = False,
             fast_limit: int = 50):
    """
    Validate for one epoch.

    [FIX-12] Dice and HD are both evaluated at the SAME threshold (best_t
             found by sweep), making the composite score coherent.
             Previously Dice used 0.3 and HD used 0.5.

    [FIX-13] HD computed only when compute_hd=True (every HD_EVAL_EVERY
             epochs) and on at most HD_EVAL_BATCHES batches, reducing the
             ~23 000 scipy calls per epoch to ~1 200 calls every 5 epochs.

    [FIX-15] All val logits are cached in CPU numpy arrays and passed to
             sweep_threshold() so the best operating threshold is found
             each epoch, addressing the Dice oscillation seen at ep30-37.
    """
    model.eval()
    total_loss   = 0.0
    n_batches    = 0
    logits_cache = []   # CPU numpy, for threshold sweep  [FIX-15]
    masks_cache  = []
    hd_logits    = []   # subset for HD computation        [FIX-13]
    hd_masks     = []

    empty_gt_count   = 0
    empty_pred_count = 0
    total_slices     = 0

    for batch_idx, (imgs, masks) in enumerate(
            tqdm(loader, desc="Val", leave=False)):
        if fast and batch_idx >= fast_limit:
            break

        imgs  = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss, _ = criterion(logits, masks)
        total_loss += loss.item()
        n_batches  += 1

        # Cache logits + masks as CPU numpy for threshold sweep  [FIX-15]
        logits_np = logits.cpu().numpy()
        masks_np  = masks.cpu().numpy()
        logits_cache.append(logits_np)
        masks_cache.append(masks_np)

        # Collect a subset for HD  [FIX-13]
        if compute_hd and len(hd_logits) < HD_EVAL_BATCHES:
            hd_logits.append(logits_np)
            hd_masks.append(masks_np)

        # Empty-slice diagnostics (using default 0.5 threshold)
        preds_diag = (torch.sigmoid(logits) > 0.5).float()
        B = masks.shape[0]
        total_slices     += B
        empty_gt_count   += (masks.sum(dim=(1, 2, 3)) == 0).sum().item()
        empty_pred_count += (preds_diag.sum(dim=(1, 2, 3)) == 0).sum().item()

        del logits, preds_diag
        if device == "cuda":
            torch.cuda.empty_cache()

    if n_batches == 0:
        return 0.0, {}

    avg_loss = total_loss / n_batches

    # ── [FIX-15] Threshold sweep ──────────────────────────────────────
    best_t, best_dice_mean, sweep_results = sweep_threshold(
        logits_cache, masks_cache
    )
    # Use best threshold for final metrics
    best_metrics = sweep_results[best_t]

    avg_metrics = {
        "best_threshold"   : best_t,
        "dice_large_bowel" : best_metrics.get("dice_large_bowel", float("nan")),
        "dice_small_bowel" : best_metrics.get("dice_small_bowel", float("nan")),
        "dice_stomach"     : best_metrics.get("dice_stomach",     float("nan")),
        "dice_mean"        : best_metrics.get("dice_mean",        float("nan")),
    }

    # ── [FIX-13] HD on subset, same threshold as Dice  [FIX-12] ──────
    avg_metrics["hausdorff_large_bowel"] = float("nan")
    avg_metrics["hausdorff_small_bowel"] = float("nan")
    avg_metrics["hausdorff_stomach"]     = float("nan")
    avg_metrics["hausdorff_mean"]        = float("nan")

    if compute_hd and hd_logits:
        all_hd = {c: [] for c in CLASS_NAMES}
        for logits_np, masks_np in zip(hd_logits, hd_masks):
            # [FIX-12] Use best_t for HD binarisation, not hardcoded 0.5
            probs_np  = 1.0 / (1.0 + np.exp(-logits_np))
            binary_np = (probs_np > best_t).astype(np.uint8)
            masks_bin = masks_np.astype(np.uint8)

            for b in range(binary_np.shape[0]):
                for c_idx, cname in enumerate(CLASS_NAMES):
                    hd = hausdorff_distance_2d(
                        binary_np[b, c_idx], masks_bin[b, c_idx]
                    )
                    # [FIX-11] None = both-empty → skip
                    if hd is not None:
                        all_hd[cname].append(hd)

        valid_hds = []
        for c_idx, cname in enumerate(CLASS_NAMES):
            vals = all_hd[cname]
            if vals:
                v = float(np.mean(vals))
                avg_metrics[f"hausdorff_{cname}"] = v
                valid_hds.append(v)
        if valid_hds:
            avg_metrics["hausdorff_mean"] = float(np.mean(valid_hds))

    # ── Composite ─────────────────────────────────────────────────────
    dm = avg_metrics["dice_mean"]
    hm = avg_metrics["hausdorff_mean"]
    if not np.isnan(dm) and not np.isnan(hm):
        avg_metrics["composite"] = 0.4 * dm + 0.6 * (1.0 - hm)
    elif not np.isnan(dm):
        # Fall back to Dice-only when HD not computed this epoch
        avg_metrics["composite"] = dm
    else:
        avg_metrics["composite"] = float("nan")

    # ── Diagnostics ───────────────────────────────────────────────────
    empty_gt_pct   = empty_gt_count   / max(total_slices, 1) * 100
    empty_pred_pct = empty_pred_count / max(total_slices, 1) * 100
    valid_pairs    = [
        sum(
            1 for lnp, mnp in zip(logits_cache, masks_cache)
            for b in range(lnp.shape[0])
            if not (mnp[b, c].sum() == 0 and
                    ((1.0 / (1.0 + np.exp(-lnp[b, c]))) > best_t).sum() == 0)
        )
        for c in range(len(CLASS_NAMES))
    ]
    print(
        f"  [Val diag] Empty GT: {empty_gt_pct:.1f}%  "
        f"| Empty pred: {empty_pred_pct:.1f}%  "
        f"| Best threshold: {best_t:.2f}  "
        f"| Valid Dice pairs: {valid_pairs}"
    )

    # Log per-threshold sweep to give visibility into distribution shift
    sweep_line = "  [Thresh sweep] " + "  ".join(
        f"{t:.2f}→{sweep_results[t]['dice_mean']:.3f}"
        for t in THRESHOLD_SWEEP
        if not np.isnan(sweep_results[t]["dice_mean"])
    )
    print(sweep_line)

    return avg_loss, avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = build_run_config(args.n_slices, args.fast, args.upscale)

    device = cfg["device"]
    print(f"\n{'='*60}")
    print(f"  GI Tract Segmentation — {cfg['n_slices']}-slice experiment")
    print(f"  Device       : {device}")
    print(f"  Encoder LR   : {cfg['lr']:.2e}")
    print(f"  Decoder LR   : {cfg['lr']*10:.2e}  (10× encoder)")
    print(f"  Warmup        : {cfg['warmup_epochs']} epochs → cosine decay")
    print(f"  Threshold     : swept {THRESHOLD_SWEEP} each epoch  [FIX-15]")
    print(f"  HD eval       : every {HD_EVAL_EVERY} epochs "
          f"on {HD_EVAL_BATCHES} batches  [FIX-13]")
    print(f"  num_workers  : {cfg['num_workers']}")
    print(f"  Checkpoint   : {cfg['checkpoint']}")
    print(f"  TensorBoard  : runs/{cfg['tb_run_name']}")
    print(f"{'='*60}\n")

    vis_dir = Path(cfg["vis_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=f"runs/{cfg['tb_run_name']}")

    # ── [FIX-5] Warn if calcStats.json looks stale ────────────────────
    if os.path.exists(cfg["stats_path"]):
        with open(cfg["stats_path"]) as f:
            stored = json.load(f)
        if stored.get("mean", 1.0) < 0.05:
            print(f"  [WARNING] calcStats.json mean is suspiciously low "
                  f"({stored['mean']:.4f}).")
            print("  Stats may have been computed before CLAHE was added.")
            print("  Delete calcStats.json and rerun to recompute.\n")
    else:
        print("calcStats.json not found — computing dataset statistics …")
        tmp_ds = GITractDataset(
            cfg["csv_path"], cfg["folder_path"],
            img_size=cfg["img_size"], stats_path=None,
        )
        from torch.utils.data import DataLoader as DL
        tmp_loader = DL(tmp_ds, batch_size=16,
                        num_workers=cfg["num_workers"])
        calc_stats(tmp_loader, cfg["stats_path"])
        print("Stats saved.\n")

    # ── DataLoaders ───────────────────────────────────────────────────
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

    if args.fast:
        from itertools import islice
        train_loader = list(islice(train_loader, 200))
        val_loader   = list(islice(val_loader,  50))
        print("  [fast mode] capped to 200 train / 50 val batches\n")

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Fixed validation sample for per-epoch visualisation ───────────
    vis_img, vis_mask = None, None
    for sample_imgs, sample_masks in val_loader:
        for b in range(sample_imgs.shape[0]):
            if sample_masks[b].sum() > 0:
                vis_img  = sample_imgs[b]
                vis_mask = sample_masks[b]
                break
        if vis_img is not None:
            break

    # ── Per-class frequency weights ───────────────────────────────────
    class_weights = (compute_class_weights(train_loader, 3, device)
                     if not args.fast
                     else torch.ones(3, device=device))

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(
        n_slices   = cfg["n_slices"],
        n_classes  = 3,
        pretrained = True,
        device     = device,
    )

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = WeightedPerClassCombinedLoss(
        class_weights=class_weights, bce_weight=0.5, dice_weight=0.5
    )

    # ── [FIX-4] Differential learning rates ──────────────────────────
    encoder_params  = list(model.model.encoder.parameters())
    decoder_params  = list(model.model.decoder.parameters())
    seg_head_params = list(model.model.segmentation_head.parameters())

    optimizer = optim.AdamW([
        {"params": encoder_params,  "lr": cfg["lr"],      "name": "encoder"},
        {"params": decoder_params,  "lr": cfg["lr"] * 10, "name": "decoder"},
        {"params": seg_head_params, "lr": cfg["lr"] * 10, "name": "seg_head"},
    ], weight_decay=1e-5)

    print(f"  Encoder params : {sum(p.numel() for p in encoder_params):,}  "
          f"lr={cfg['lr']:.2e}")
    print(f"  Decoder params : {sum(p.numel() for p in decoder_params):,}  "
          f"lr={cfg['lr']*10:.2e}")
    print(f"  Seg-head params: {sum(p.numel() for p in seg_head_params):,}  "
          f"lr={cfg['lr']*10:.2e}\n")

    # ── Scheduler: linear warmup → cosine annealing ───────────────────
    warmup_epochs = cfg["warmup_epochs"]
    cosine_epochs = cfg["epochs"] - warmup_epochs

    warmup_sched = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(cosine_epochs, 1), eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    # ── AMP scaler ────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 1
    best_dice   = 0.0
    global_step = 0
    log_history = []
    best_threshold = 0.3   # updated each epoch by sweep

    if args.resume and os.path.exists(cfg["crash_checkpoint"]):
        print(f"Resuming from crash checkpoint: {cfg['crash_checkpoint']}")
        ckpt = torch.load(cfg["crash_checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch    = ckpt["epoch"] + 1
        best_dice      = ckpt.get("best_dice",      0.0)
        global_step    = ckpt.get("global_step",    0)
        log_history    = ckpt.get("log_history",    [])
        best_threshold = ckpt.get("best_threshold", 0.3)
        # [FIX-10] Restore scaler state
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"  Resuming from epoch {start_epoch}  "
              f"(best Dice={best_dice:.4f}, "
              f"best threshold={best_threshold:.2f})\n")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        current_lr  = scheduler.get_last_lr()[0]
        do_hd       = (epoch % HD_EVAL_EVERY == 0) and not args.fast
        print(f"\nEpoch {epoch}/{cfg['epochs']}  lr={current_lr:.2e}"
              + ("  [+HD]" if do_hd else ""))

        # [FIX-9] Removed no-op sampler generator reassignment.
        # The DataLoader captures the generator reference at construction
        # time; mutating sampler.generator afterwards has no effect.
        # Epoch-level randomness comes naturally from PyTorch's default RNG.

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, writer, global_step,
            epoch=epoch, fast=args.fast,
            accum_steps=cfg["accum_steps"],
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device,
            fast=args.fast, epoch=epoch, compute_hd=do_hd,
        )
        scheduler.step()

        if not val_metrics:
            continue

        # Track best threshold across epochs  [FIX-15]
        best_threshold = val_metrics.get("best_threshold", best_threshold)

        dice_mean = val_metrics.get("dice_mean",      float("nan"))
        hd_mean   = val_metrics.get("hausdorff_mean", float("nan"))
        composite = val_metrics.get("composite",      float("nan"))

        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "nan"

        hd_suffix = (
            f"\n  HD    — large_bowel: "
            f"{fmt(val_metrics.get('hausdorff_large_bowel', float('nan')))}  "
            f"small_bowel: "
            f"{fmt(val_metrics.get('hausdorff_small_bowel', float('nan')))}  "
            f"stomach: "
            f"{fmt(val_metrics.get('hausdorff_stomach', float('nan')))}  "
            f"mean: {fmt(hd_mean)}"
            if do_hd else ""
        )

        print(
            f"  Train Loss : {train_loss:.4f}  |  Val Loss : {val_loss:.4f}\n"
            f"  Dice  — large_bowel: "
            f"{fmt(val_metrics.get('dice_large_bowel', float('nan')))}  "
            f"small_bowel: "
            f"{fmt(val_metrics.get('dice_small_bowel', float('nan')))}  "
            f"stomach: "
            f"{fmt(val_metrics.get('dice_stomach', float('nan')))}  "
            f"mean: {fmt(dice_mean)}"
            + hd_suffix +
            f"\n  Composite  : {fmt(composite)}"
            f"  (threshold={best_threshold:.2f})"
        )

        # ── TensorBoard ───────────────────────────────────────────────
        writer.add_scalar("Epoch/train_loss",   train_loss,     epoch)
        writer.add_scalar("Epoch/val_loss",     val_loss,       epoch)
        writer.add_scalar("Epoch/lr",           current_lr,     epoch)
        writer.add_scalar("Epoch/best_threshold", best_threshold, epoch)

        for key, tag in [
            ("dice_mean",      "Epoch/dice_mean"),
            ("hausdorff_mean", "Epoch/hd_mean"),
            ("composite",      "Epoch/composite"),
        ]:
            v = val_metrics.get(key, float("nan"))
            if not np.isnan(v):
                writer.add_scalar(tag, v, epoch)

        for c in CLASS_NAMES:
            for prefix, tb_prefix in [("dice_", "Epoch/dice_"),
                                       ("hausdorff_", "Epoch/hd_")]:
                v = val_metrics.get(f"{prefix}{c}", float("nan"))
                if not np.isnan(v):
                    writer.add_scalar(f"{tb_prefix}{c}", v, epoch)

        # Log per-threshold Dice sweep to TensorBoard  [FIX-15]
        # (stored in val_metrics as threshold_sweep sub-dict isn't feasible
        # here so we just log the best and rely on the console output above)

        # ── Mask overlay PNG ──────────────────────────────────────────
        if vis_img is not None:
            mask_path = vis_dir / f"epoch_{epoch:04d}.png"
            save_mask_overlay(
                model, vis_img, vis_mask,
                str(mask_path), epoch, device,
                threshold=best_threshold,   # use calibrated threshold
            )
            print(f"  Mask overlay → {mask_path}")
            try:
                from PIL import Image as PILImage
                img_pil = PILImage.open(str(mask_path)).convert("RGB")
                img_np  = np.array(img_pil).transpose(2, 0, 1) / 255.0
                writer.add_image(
                    "Segmentation/overlay",
                    torch.tensor(img_np, dtype=torch.float32),
                    epoch,
                )
            except Exception:
                pass

        # ── Best model checkpoint ─────────────────────────────────────
        if not np.isnan(dice_mean) and dice_mean > best_dice:
            best_dice = dice_mean
            torch.save({
                "epoch"         : epoch,
                "model_state"   : model.state_dict(),
                "optimizer"     : optimizer.state_dict(),
                "dice_mean"     : dice_mean,
                "best_threshold": best_threshold,
                "config"        : cfg,
            }, cfg["checkpoint"])
            print(f"  ✓ Saved best model (Dice={best_dice:.4f}, "
                  f"threshold={best_threshold:.2f}) → {cfg['checkpoint']}")

        # ── Crash-recovery checkpoint ─────────────────────────────────
        torch.save({
            "epoch"         : epoch,
            "model_state"   : model.state_dict(),
            "optimizer"     : optimizer.state_dict(),
            "scheduler"     : scheduler.state_dict(),
            # [FIX-10] Save scaler state so AMP resumes correctly
            "scaler"        : scaler.state_dict() if scaler is not None else None,
            "best_dice"     : best_dice,
            "best_threshold": best_threshold,
            "global_step"   : global_step,
            "log_history"   : log_history,
            "config"        : cfg,
        }, cfg["crash_checkpoint"])

        # ── JSON log ──────────────────────────────────────────────────
        log_history.append({
            "epoch"         : epoch,
            "train_loss"    : train_loss,
            "val_loss"      : val_loss,
            "best_threshold": best_threshold,
            **{k: (float(v) if not np.isnan(float(v)) else None)
               for k, v in val_metrics.items()
               if isinstance(v, (int, float))},
        })
        with open(cfg["log_file"], "w") as f:
            json.dump(log_history, f, indent=2)

    writer.close()
    print(f"\nTraining complete.")
    print(f"  Best Dice      : {best_dice:.4f}")
    print(f"  Best threshold : {best_threshold:.2f}")
    print(f"  Best checkpoint: {cfg['checkpoint']}")
    print(f"  Log            : {cfg['log_file']}")
    print(f"  TensorBoard    : tensorboard --logdir runs/")
    print(f"  Mask overlays  : {cfg['vis_dir']}/epoch_XXXX.png")


if __name__ == "__main__":
    main()