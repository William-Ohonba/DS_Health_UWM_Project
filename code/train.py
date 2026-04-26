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
  [FIX-8]  Small bowel class weight boosted x1.5 after frequency calculation
  [FIX-9]  Sampler generator reassignment removed — it was a no-op
  [FIX-10] GradScaler state saved + restored in crash checkpoint
  [FIX-11] hausdorff_distance_2d returns None for both-empty pairs (loss.py)
  [FIX-12] Dice and HD now both evaluated at the SAME threshold (0.3)
  [FIX-13] HD computed only every HD_EVAL_EVERY epochs (default 5) and on
           a random subsample of HD_EVAL_BATCHES batches (default 40).
  [FIX-14] save_mask_overlay wraps body in try/finally
  [FIX-15] Threshold sweep: after every epoch
  [FIX-16] validate(): logits_cache capped at SWEEP_CACHE_BATCHES (200)
  [FIX-17] train_one_epoch(): AMP branch now unpacks _forward() into `loss`
  [FIX-18] augmentations.py: alpha_affine removed, shift_limit split
  [FIX-19] Best-model checkpoint saved on composite score when HD available
  [FIX-20] valid_pairs diagnostic reads pre-computed counts
  [FIX-21] ReduceLROnPlateau added as secondary scheduler
  [FIX-22] get_dataloaders() pin_memory threaded through from cfg
  [FIX-23] Encoder LR raised from 3e-5 → 1e-4; LR ratio reduced from 10x to 3x
  [FIX-24] SequentialLR + ReduceLROnPlateau conflict resolved.
  [FIX-25] dataset.py: WeightedRandomSampler weight 5.0 → 2.0
  [FIX-26] loss.py: smooth=1.0 in metric (not loss) for small structures
  [FIX-27] model.py: Dropout2d(0.2) before segmentation head
  [FIX-28] (same as FIX-24, cross-reference)
  [FIX-29] best_score comparison normalised: dice_mean is always primary metric
  [FIX-30] Auxiliary presence detection head added to model.py
  [FIX-31] model.py: presence_head operates on detached encoder features
  [FIX-32] loss.py: Dice smooth raised 1e-6 → 1e-4
  [FIX-33] Plateau scheduler now monitors negated dice_mean, not val_loss
  [FIX-34] PRESENCE_LOSS_WEIGHT annealed 0.0 → 0.3 over first 10 epochs
  [FIX-35] Gradient norm logged to TensorBoard every update step
  [FIX-36] Early stopping with best-weight restoration
  [FIX-37] Threshold sweep lower bound raised 0.20 → 0.25
  [FIX-38] Gradient accumulation boundary condition fixed
  [FIX-39] LR reduced: encoder 1e-4 → 5e-5, decoder ratio 3x → 2x
  [FIX-40] Plateau patience 7 → 3, factor 0.5 → 0.4
  [FIX-41] Weight decay raised 1e-5 → 1e-3
  [FIX-42] Early stop patience 15 → 20, min_epochs guard added
  [FIX-43] EMA smoothing on val dice before feeding plateau scheduler
  [FIX-44] Encoder frozen for first FREEZE_ENCODER_EPOCHS epochs
  [FIX-45] Early-stop counter and best_dice reset after encoder unfreeze
  [FIX-46] EMA state reset at encoder unfreeze boundary
  [FIX-47] Plateau scheduler not stepped during encoder-frozen epochs
  [FIX-48] Early stop patience raised 20 → 35, min_epochs raised 15 → 25.
           Root cause of premature termination at epoch 49: the model's
           best dice (0.321 at epoch 29) was followed by 20 consecutive
           epochs all within 0.01 of that best, which is normal noise for
           this dataset at a nearly-converged LR. Patience=20 treated this
           as a true plateau and stopped training while the model was
           actively producing competitive scores. Raising to 35 gives the
           model enough runway to benefit from LR reductions and find a
           new best.
  [FIX-49] Plateau patience raised 3 → 8, factor 0.4 → 0.5.
           With patience=3 on EMA-smoothed neg_dice, the LR was reduced
           every 3-4 epochs of flat EMA, crashing to 1.28e-06 by epoch 44
           (effectively zero — min_lr=1e-7 is one more halving away).
           At lr=1.28e-06 the model cannot escape any local optimum
           regardless of how much patience the early stopping gives it.
           patience=8 means the LR only drops after 8 truly flat epochs
           (not 3 noisy ones), and factor=0.5 (was 0.4) makes each drop
           less aggressive so more reductions remain useful before hitting
           min_lr.
  [FIX-50] EMA alpha raised 0.3 → 0.6.
           With alpha=0.3, the EMA had a time constant of ~3 epochs,
           meaning it was still heavily influenced by epochs 3-5 epochs
           ago. This caused the plateau scheduler to see a signal that
           lagged the actual dice by several epochs, triggering reductions
           right when dice was genuinely improving. alpha=0.6 tracks the
           current dice more closely (time constant ~1.7 epochs) while
           still filtering single-epoch noise.
  [FIX-51] min_lr raised 1e-7 → 5e-6.
           At lr < 5e-6 the effective weight update per step is smaller
           than numerical precision for float16 gradients and provides no
           meaningful optimisation. Stopping plateau reductions at 5e-6
           prevents the model from entering a permanently frozen state
           while still allowing significant LR reduction from the initial
           5e-5 encoder / 1e-4 decoder rates.

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
import argparse
import multiprocessing
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
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

HD_EVAL_EVERY   = 1
HD_EVAL_BATCHES = 80

# [FIX-37] Lower bound raised 0.20 → 0.25
THRESHOLD_SWEEP = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

SWEEP_CACHE_BATCHES = 99_999

PRESENCE_LOSS_WEIGHT   = 0.3
PRESENCE_ANNEAL_EPOCHS = 10   # [FIX-34]

# [FIX-48] Raised from 20 → 35; min_epochs raised from 15 → 25
EARLY_STOP_PATIENCE = 35
MIN_EPOCHS          = 25

# [FIX-44]
FREEZE_ENCODER_EPOCHS = 3

# [FIX-50] Raised from 0.3 → 0.6 for tighter tracking of current dice
EMA_ALPHA = 0.6


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
    "accum_steps"  : 2,
    "epochs"       : 100,
    # [FIX-39] encoder LR 5e-5, decoder 2x
    "lr"           : 5e-5,
    "lr_ratio"     : 2,
    "warmup_epochs": 5,
    "device"       : _DEVICE,
    "pin_memory"   : _DEVICE == "cuda",
}


# ---------------------------------------------------------------------------
# Presence loss weight annealing  [FIX-34]
# ---------------------------------------------------------------------------

def get_presence_weight(epoch: int) -> float:
    """Linearly ramp 0 → PRESENCE_LOSS_WEIGHT over PRESENCE_ANNEAL_EPOCHS."""
    if epoch >= PRESENCE_ANNEAL_EPOCHS:
        return PRESENCE_LOSS_WEIGHT
    return PRESENCE_LOSS_WEIGHT * (epoch / PRESENCE_ANNEAL_EPOCHS)

def get_hd_loss_weight(epoch: int) -> float:
    """[FIX-52] Linearly ramp 0 → HD_LOSS_WEIGHT over HD_LOSS_ANNEAL_EPOCHS."""
    if epoch >= HD_LOSS_ANNEAL_EPOCHS:
        return HD_LOSS_WEIGHT
    return HD_LOSS_WEIGHT * (epoch / HD_LOSS_ANNEAL_EPOCHS)
 
 
# ---------------------------------------------------------------------------
# Boundary (soft-HD) loss  [FIX-52]
# ---------------------------------------------------------------------------
 
# 3x3 Laplacian kernel for edge detection
_LAPLACIAN = torch.tensor(
    [[0.,  1., 0.],
     [1., -4., 1.],
     [0.,  1., 0.]], dtype=torch.float32
).view(1, 1, 3, 3)
 
 
def boundary_loss(seg_logits: torch.Tensor,
                  targets: torch.Tensor) -> torch.Tensor:
    """
    Soft boundary loss: 1 - F1(pred_edges, gt_edges).
    Edges extracted via Laplacian on sigmoid probs (pred) and float targets.
    Operates per-class, returns scalar mean.
 
    [FIX-52] This gives the model a direct gradient signal for boundary
    precision, which pure Dice/BCE lack. Targeting the HD gap.
    """
    B, C, H, W = seg_logits.shape
    device = seg_logits.device
 
    kernel = _LAPLACIAN.to(device)  # [1,1,3,3]
    # Expand kernel for C-channel grouped conv
    kernel_c = kernel.expand(C, 1, 3, 3)  # [C,1,3,3]
 
    probs = torch.sigmoid(seg_logits)  # [B,C,H,W]
 
    # Extract edges via Laplacian magnitude, clamp to [0,1]
    pred_edges = F.conv2d(probs,    kernel_c, padding=1, groups=C).abs().clamp(0, 1)
    gt_edges   = F.conv2d(targets,  kernel_c, padding=1, groups=C).abs().clamp(0, 1)
 
    smooth = 1e-6
    inter  = (pred_edges * gt_edges).sum(dim=(2, 3))          # [B,C]
    denom  = pred_edges.sum(dim=(2, 3)) + gt_edges.sum(dim=(2, 3))  # [B,C]
    f1     = (2.0 * inter + smooth) / (denom + smooth)        # [B,C]
 
    return 1.0 - f1.mean()
 


# ---------------------------------------------------------------------------
# Per-class frequency weighting
# ---------------------------------------------------------------------------

def compute_class_weights(loader, n_classes: int = 3,
                          device: str = "cpu") -> torch.Tensor:
    """Inverse-frequency weights; small bowel boosted x1.5 [FIX-8]."""
    print("Computing per-class pixel frequencies for loss weighting ...")
    class_pixel_sum = np.zeros(n_classes, dtype=np.float64)
    total_pixels    = 0

    for _, masks in tqdm(loader, desc="Class freq scan", leave=False):
        for c in range(n_classes):
            class_pixel_sum[c] += masks[:, c].sum().item()
        total_pixels += masks[:, 0].numel() * masks.shape[0]

    freq    = class_pixel_sum / (total_pixels + 1e-8)
    weights = 1.0 / (freq + 1e-4)
    weights[IDX_SMALL_BOWEL] *= 1.5
    weights = weights / weights.sum() * n_classes

    print(f"  Class frequencies : {freq}")
    print(f"  Class loss weights: {weights}  (small_bowel boosted x1.5)")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Weighted per-class loss
# ---------------------------------------------------------------------------

class WeightedPerClassCombinedLoss(torch.nn.Module):
    """BCE + Dice per class, combined with inverse-frequency weights."""

    def __init__(self, class_weights: torch.Tensor,
                 bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.bce_w  = bce_weight
        self.dice_w = dice_weight
        self.bce    = torch.nn.BCEWithLogitsLoss(reduction="none")

    def _dice_loss(self, logits, targets, smooth: float = 1e-4):
        # [FIX-32] smooth=1e-4 prevents zero-gradient trap on empty slices
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice  = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
        return dice.mean(dim=0)  # [C]

    def forward(self, logits, targets):
        bce_per_class  = self.bce(logits, targets).mean(dim=(0, 2, 3))
        dice_per_class = self._dice_loss(logits, targets)
        loss_per_class = self.bce_w * bce_per_class + self.dice_w * dice_per_class
        weighted_loss  = (loss_per_class * self.class_weights).sum() \
                       / self.class_weights.sum()
        return weighted_loss, loss_per_class.detach()


# ---------------------------------------------------------------------------
# Dice metric with threshold sweep
# ---------------------------------------------------------------------------

def compute_dice_at_threshold(logits_list, masks_list, threshold, smooth=1.0):
    """Per-class Dice; skips both-empty pairs; smooth=1.0 in metric [FIX-26]."""
    per_class    = {c: [] for c in CLASS_NAMES}
    valid_counts = {c: 0  for c in CLASS_NAMES}

    for logits_np, masks_np in zip(logits_list, masks_list):
        probs = 1.0 / (1.0 + np.exp(-logits_np))
        preds = (probs > threshold).astype(np.float32)

        for c, cname in enumerate(CLASS_NAMES):
            for b in range(logits_np.shape[0]):
                gt_sum   = masks_np[b, c].sum()
                pred_sum = preds[b, c].sum()
                if gt_sum == 0 and pred_sum == 0:
                    continue
                valid_counts[cname] += 1
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

    result["dice_mean"]    = float(np.mean(valid_means)) if valid_means else float("nan")
    result["valid_counts"] = valid_counts
    return result


def sweep_threshold(logits_list, masks_list):
    """[FIX-37] Sweep from 0.25; return (best_t, best_dice, all_results)."""
    best_t      = THRESHOLD_SWEEP[0]
    best_dice   = -1.0
    all_results = {}

    for t in THRESHOLD_SWEEP:
        res = compute_dice_at_threshold(logits_list, masks_list, t)
        dm  = res["dice_mean"]
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
                      save_path, epoch, device, threshold=0.5):
    """[FIX-14] try/finally ensures model.train() is always restored."""
    model.eval()
    try:
        img_t = sample_img.unsqueeze(0).to(device)
        logits, _ = model(img_t)
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
        fig.suptitle(f"Epoch {epoch} - Segmentation Preview",
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
        model.train()


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


def build_run_config(n_slices, fast, upscale):
    cfg              = BASE_CONFIG.copy()
    cfg["n_slices"]  = n_slices
    cfg["img_size"]  = 448 if upscale else 320
    cfg["num_workers"] = optimal_workers(_DEVICE)
    suffix           = "_448" if upscale else "_320"
    cfg["checkpoint"]       = f"best_model_{n_slices}slice{suffix}.pth"
    cfg["crash_checkpoint"] = f"crash_recovery_{n_slices}slice.pth"
    cfg["log_file"]         = f"training_log_{n_slices}slice.json"
    cfg["tb_run_name"]      = f"gi_{n_slices}slice"
    cfg["vis_dir"]          = f"vis_{n_slices}slice"
    if fast:
        cfg["epochs"]           = 5
        cfg["checkpoint"]       = f"fast_model_{n_slices}slice.pth"
        cfg["crash_checkpoint"] = f"fast_crash_{n_slices}slice.pth"
        cfg["log_file"]         = f"fast_log_{n_slices}slice.json"
        cfg["tb_run_name"]      = f"gi_fast_{n_slices}slice"
        cfg["vis_dir"]          = f"vis_fast_{n_slices}slice"
    return cfg


# ---------------------------------------------------------------------------
# Encoder freeze / unfreeze  [FIX-44]
# ---------------------------------------------------------------------------

def set_encoder_trainable(model, trainable: bool):
    """Freeze or unfreeze the SMP encoder parameters."""
    for p in model.model.encoder.parameters():
        p.requires_grad = trainable
    state = "unfrozen" if trainable else "frozen"
    print(f"  [FIX-44] Encoder {state}.")


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion,
                    device, scaler, writer, global_step,
                    epoch, fast, accum_steps=2, fast_limit=200):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    presence_weight = get_presence_weight(epoch)  # [FIX-34]

    pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)

    # [FIX-38] Clean window-based accumulation
    optimizer.zero_grad()
    accum_count = 0

    for batch_idx, (imgs, masks) in enumerate(pbar):
        if fast and batch_idx >= fast_limit:
            break

        imgs  = imgs.to(device)
        masks = masks.to(device)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, per_class = _forward(
                    model, criterion, imgs, masks, presence_weight
                )
            scaler.scale(loss / accum_steps).backward()
        else:
            loss, per_class = _forward(
                model, criterion, imgs, masks, presence_weight
            )
            (loss / accum_steps).backward()

        accum_count += 1
        is_last_batch = (batch_idx + 1 == len(loader)) if not fast \
                        else (batch_idx + 1 >= fast_limit)
        should_update = (accum_count == accum_steps) or is_last_batch

        if should_update:
            if scaler is not None:
                scaler.unscale_(optimizer)

            # [FIX-35] Log pre-clip gradient norm
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            writer.add_scalar("Batch/grad_norm", grad_norm, global_step)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            accum_count = 0

        loss_val    = loss.item()
        total_loss += loss_val
        n_batches  += 1
        global_step += 1

        writer.add_scalar("Batch/train_loss", loss_val, global_step)
        for c, cname in enumerate(CLASS_NAMES):
            writer.add_scalar(f"Batch/loss_{cname}", per_class[c].item(), global_step)

        pbar.set_postfix(loss=f"{loss_val:.4f}", pw=f"{presence_weight:.2f}")

    return total_loss / max(n_batches, 1), global_step


def _forward(model, criterion, imgs, masks, presence_weight=0.3):
    """Seg loss + annealed presence loss [FIX-34]. Presence detached [FIX-31]."""
    seg_logits, presence_logits = model(imgs)
    seg_loss, per_class = criterion(seg_logits, masks)

    presence_labels = (masks.sum(dim=(2, 3)) > 0).float()
    presence_loss   = F.binary_cross_entropy_with_logits(
        presence_logits, presence_labels
    )
    loss = seg_loss + presence_weight * presence_loss
    return loss, per_class


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device, fast,
             epoch=1, compute_hd=False, fast_limit=50,
             presence_weight=0.3):
    """[FIX-33] Returns neg_dice for plateau. [FIX-34] annealed weight."""
    model.eval()
    total_loss   = 0.0
    n_batches    = 0
    logits_cache = []
    masks_cache  = []
    hd_logits    = []
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

        seg_logits, presence_logits = model(imgs)
        seg_loss, _ = criterion(seg_logits, masks)

        presence_labels_v = (masks.sum(dim=(2, 3)) > 0).float()
        presence_loss_v   = F.binary_cross_entropy_with_logits(
            presence_logits, presence_labels_v
        )
        combined_loss = seg_loss + presence_weight * presence_loss_v

        total_loss += combined_loss.item()
        n_batches  += 1

        logits_np = seg_logits.cpu().numpy()
        masks_np  = masks.cpu().numpy()

        if len(logits_cache) < SWEEP_CACHE_BATCHES:
            logits_cache.append(logits_np)
            masks_cache.append(masks_np)

        if compute_hd and len(hd_logits) < HD_EVAL_BATCHES:
            hd_logits.append(logits_np)
            hd_masks.append(masks_np)

        preds_diag = (torch.sigmoid(seg_logits) > 0.5).float()
        B = masks.shape[0]
        total_slices     += B
        empty_gt_count   += (masks.sum(dim=(1, 2, 3)) == 0).sum().item()
        empty_pred_count += (preds_diag.sum(dim=(1, 2, 3)) == 0).sum().item()

        del seg_logits, presence_logits, presence_labels_v, preds_diag
        if device == "cuda":
            torch.cuda.empty_cache()

    if n_batches == 0:
        return 0.0, float("nan"), {}

    avg_loss = total_loss / n_batches

    best_t, best_dice_mean, sweep_results = sweep_threshold(
        logits_cache, masks_cache
    )
    best_metrics = sweep_results[best_t]
    valid_counts = best_metrics.get("valid_counts", {})
    valid_pairs  = [valid_counts.get(cname, 0) for cname in CLASS_NAMES]

    avg_metrics = {
        "best_threshold"   : best_t,
        "dice_large_bowel" : best_metrics.get("dice_large_bowel", float("nan")),
        "dice_small_bowel" : best_metrics.get("dice_small_bowel", float("nan")),
        "dice_stomach"     : best_metrics.get("dice_stomach",     float("nan")),
        "dice_mean"        : best_metrics.get("dice_mean",        float("nan")),
        "valid_counts"     : valid_counts,
    }

    for k in ["hausdorff_large_bowel", "hausdorff_small_bowel",
              "hausdorff_stomach", "hausdorff_mean"]:
        avg_metrics[k] = float("nan")

    if compute_hd and hd_logits:
        all_hd = {c: [] for c in CLASS_NAMES}
        for logits_np, masks_np in zip(hd_logits, hd_masks):
            probs_np  = 1.0 / (1.0 + np.exp(-logits_np))
            binary_np = (probs_np > best_t).astype(np.uint8)
            masks_bin = masks_np.astype(np.uint8)
            for b in range(binary_np.shape[0]):
                for c_idx, cname in enumerate(CLASS_NAMES):
                    hd = hausdorff_distance_2d(
                        binary_np[b, c_idx], masks_bin[b, c_idx]
                    )
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

    dm = avg_metrics["dice_mean"]
    hm = avg_metrics["hausdorff_mean"]
    if not np.isnan(dm) and not np.isnan(hm):
        avg_metrics["composite"] = 0.4 * dm + 0.6 * (1.0 - hm)
    elif not np.isnan(dm):
        avg_metrics["composite"] = dm
    else:
        avg_metrics["composite"] = float("nan")

    empty_gt_pct   = empty_gt_count   / max(total_slices, 1) * 100
    empty_pred_pct = empty_pred_count / max(total_slices, 1) * 100
    print(
        f"  [Val diag] Empty GT: {empty_gt_pct:.1f}%  "
        f"| Empty pred: {empty_pred_pct:.1f}%  "
        f"| Best threshold: {best_t:.2f}  "
        f"| Valid Dice pairs: {valid_pairs}"
    )
    sweep_line = "  [Thresh sweep] " + "  ".join(
        f"{t:.2f}→{sweep_results[t]['dice_mean']:.3f}"
        for t in THRESHOLD_SWEEP
        if not np.isnan(sweep_results[t]["dice_mean"])
    )
    print(sweep_line)

    neg_dice = -dm if not np.isnan(dm) else float("nan")
    return avg_loss, neg_dice, avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = build_run_config(args.n_slices, args.fast, args.upscale)

    device   = cfg["device"]
    lr_ratio = cfg["lr_ratio"]

    print(f"\n{'='*60}")
    print(f"  GI Tract Segmentation — {cfg['n_slices']}-slice experiment")
    print(f"  Device            : {device}")
    print(f"  Encoder LR        : {cfg['lr']:.2e}")
    print(f"  Decoder LR        : {cfg['lr']*lr_ratio:.2e}  ({lr_ratio}x)")
    print(f"  Weight decay      : 1e-3  [FIX-41]")
    print(f"  Encoder freeze    : first {FREEZE_ENCODER_EPOCHS} epochs  [FIX-44]")
    print(f"  Unfreeze resets   : best_dice / counter / EMA  [FIX-45/46]")
    print(f"  Plateau gated     : no step while frozen  [FIX-47]")
    print(f"  Warmup            : {cfg['warmup_epochs']} epochs → plateau-only")
    print(f"  Plateau           : patience=8 factor=0.5 neg_dice  [FIX-49]")
    print(f"  min_lr            : 5e-6  [FIX-51]")
    print(f"  EMA alpha         : {EMA_ALPHA}  [FIX-50]")
    print(f"  Threshold sweep   : {THRESHOLD_SWEEP}  [FIX-37]")
    print(f"  Presence anneal   : 0→{PRESENCE_LOSS_WEIGHT} over "
          f"{PRESENCE_ANNEAL_EPOCHS} ep  [FIX-34]")
    print(f"  Early stop        : patience={EARLY_STOP_PATIENCE} "
          f"min_ep={MIN_EPOCHS}  [FIX-48]")
    print(f"  Checkpoint        : {cfg['checkpoint']}")
    print(f"  TensorBoard       : runs/{cfg['tb_run_name']}")
    print(f"{'='*60}\n")

    vis_dir = Path(cfg["vis_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{cfg['tb_run_name']}")

    # Stale stats warning [FIX-5]
    if os.path.exists(cfg["stats_path"]):
        with open(cfg["stats_path"]) as f:
            stored = json.load(f)
        if stored.get("mean", 1.0) < 0.05:
            print(f"  [WARNING] calcStats.json mean={stored['mean']:.4f} "
                  f"suspicious. Delete and rerun.\n")
    else:
        print("calcStats.json not found — computing ...")
        tmp_ds = GITractDataset(
            cfg["csv_path"], cfg["folder_path"],
            img_size=cfg["img_size"], stats_path=None,
        )
        from torch.utils.data import DataLoader as DL
        tmp_loader = DL(tmp_ds, batch_size=16, num_workers=cfg["num_workers"])
        calc_stats(tmp_loader, cfg["stats_path"])
        print("Stats saved.\n")

    # DataLoaders
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
        pin_memory    = cfg["pin_memory"],
    )

    if args.fast:
        from itertools import islice
        train_loader = list(islice(train_loader, 200))
        val_loader   = list(islice(val_loader,  50))
        print("  [fast mode] capped to 200 train / 50 val batches\n")

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Fixed val sample for visualisation
    vis_img, vis_mask = None, None
    for sample_imgs, sample_masks in val_loader:
        for b in range(sample_imgs.shape[0]):
            if sample_masks[b].sum() > 0:
                vis_img  = sample_imgs[b]
                vis_mask = sample_masks[b]
                break
        if vis_img is not None:
            break

    # Per-class frequency weights
    class_weights = (compute_class_weights(train_loader, 3, device)
                     if not args.fast
                     else torch.ones(3, device=device))

    # Model
    model = build_model(
        n_slices   = cfg["n_slices"],
        n_classes  = 3,
        pretrained = True,
        device     = device,
        dropout    = 0.2,
    )

    # Loss
    criterion = WeightedPerClassCombinedLoss(
        class_weights=class_weights, bce_weight=0.5, dice_weight=0.5
    )

    # Differential LR param groups [FIX-39]
    encoder_params  = list(model.model.encoder.parameters())
    decoder_params  = list(model.model.decoder.parameters())
    seg_head_params = list(model.model.segmentation_head.parameters())
    presence_params = list(model.presence_head.parameters())

    optimizer = optim.AdamW([
        {"params": encoder_params,  "lr": cfg["lr"],             "name": "encoder"},
        {"params": decoder_params,  "lr": cfg["lr"] * lr_ratio,  "name": "decoder"},
        {"params": seg_head_params, "lr": cfg["lr"] * lr_ratio,  "name": "seg_head"},
        {"params": presence_params, "lr": cfg["lr"] * lr_ratio,  "name": "presence"},
    ], weight_decay=1e-3)  # [FIX-41]

    print(f"  Encoder params  : {sum(p.numel() for p in encoder_params):,}  "
          f"lr={cfg['lr']:.2e}")
    print(f"  Decoder params  : {sum(p.numel() for p in decoder_params):,}  "
          f"lr={cfg['lr']*lr_ratio:.2e}")
    print(f"  Seg-head params : {sum(p.numel() for p in seg_head_params):,}  "
          f"lr={cfg['lr']*lr_ratio:.2e}")
    print(f"  Presence params : {sum(p.numel() for p in presence_params):,}  "
          f"lr={cfg['lr']*lr_ratio:.2e}\n")

    # Warmup scheduler [FIX-24]
    warmup_epochs = cfg["warmup_epochs"]
    warmup_sched  = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # [FIX-49] patience=8, factor=0.5 — less aggressive than before
    # [FIX-51] min_lr=5e-6 — prevents LR from reaching effectively zero
    # [FIX-33] mode="min" on neg_dice
    plateau_sched = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=5e-6,
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # ---------------------------------------------------------------------------
    # Training state
    # ---------------------------------------------------------------------------
    start_epoch       = 1
    best_dice         = 0.0
    global_step       = 0
    log_history       = []
    best_threshold    = 0.30
    warmup_done       = False
    epochs_no_improve = 0
    best_model_state  = None
    ema_neg_dice      = None   # [FIX-43]
    encoder_frozen    = False  # [FIX-44]

    if args.resume and os.path.exists(cfg["crash_checkpoint"]):
        print(f"Resuming from: {cfg['crash_checkpoint']}")
        ckpt = torch.load(cfg["crash_checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch       = ckpt["epoch"] + 1
        best_dice         = ckpt.get("best_dice", 0.0)
        global_step       = ckpt.get("global_step", 0)
        log_history       = ckpt.get("log_history", [])
        best_threshold    = ckpt.get("best_threshold", 0.30)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        ema_neg_dice      = ckpt.get("ema_neg_dice", None)
        encoder_frozen    = ckpt.get("encoder_frozen", False)
        warmup_done       = start_epoch > warmup_epochs
        if not warmup_done and "scheduler" in ckpt:
            warmup_sched.load_state_dict(ckpt["scheduler"])
        if "plateau_scheduler" in ckpt:
            plateau_sched.load_state_dict(ckpt["plateau_scheduler"])
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if encoder_frozen:
            set_encoder_trainable(model, False)
        print(f"  Resumed epoch {start_epoch}  best_dice={best_dice:.4f}  "
              f"no-improve={epochs_no_improve}  frozen={encoder_frozen}\n")

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        current_lr      = optimizer.param_groups[0]["lr"]
        do_hd           = (epoch % HD_EVAL_EVERY == 0) and not args.fast
        presence_weight = get_presence_weight(epoch)

        # [FIX-44] Freeze encoder at epoch 1
        if epoch == 1 and not encoder_frozen:
            set_encoder_trainable(model, False)
            encoder_frozen = True

        # [FIX-44/45/46] Unfreeze and reset state at boundary epoch
        if epoch == FREEZE_ENCODER_EPOCHS + 1 and encoder_frozen:
            set_encoder_trainable(model, True)
            encoder_frozen = False

            # [FIX-45] Reset early-stop state — frozen phase optimum is
            # not representative of the joint encoder+decoder optimum
            print(f"  [FIX-45] Resetting best_dice "
                  f"({best_dice:.4f}→0.0) and no-improve "
                  f"({epochs_no_improve}→0) at encoder unfreeze.")
            best_dice         = 0.0
            epochs_no_improve = 0
            best_model_state  = None

            # [FIX-46] Reset EMA so stale frozen value doesn't skew plateau
            ema_str = f"{ema_neg_dice:.4f}" if ema_neg_dice is not None else "None"
            print(f"  [FIX-46] Resetting EMA neg_dice ({ema_str}→None).")
            ema_neg_dice = None

        print(f"\nEpoch {epoch}/{cfg['epochs']}  lr={current_lr:.2e}"
              f"  presence_w={presence_weight:.3f}"
              + ("  [encoder frozen]" if encoder_frozen else "")
              + ("  [+HD]" if do_hd else ""))

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, writer, global_step,
            epoch=epoch, fast=args.fast,
            accum_steps=cfg["accum_steps"],
        )

        val_loss, neg_dice_raw, val_metrics = validate(
            model, val_loader, criterion, device,
            fast=args.fast, epoch=epoch, compute_hd=do_hd,
            presence_weight=presence_weight,
        )

        # [FIX-43/50] EMA smooth neg_dice — alpha=0.6 tracks current closely
        if not np.isnan(neg_dice_raw):
            if ema_neg_dice is None:
                ema_neg_dice = neg_dice_raw  # bias-free init
            else:
                ema_neg_dice = (EMA_ALPHA * neg_dice_raw
                                + (1 - EMA_ALPHA) * ema_neg_dice)
            neg_dice_for_plateau = ema_neg_dice
        else:
            neg_dice_for_plateau = float("nan")

        # [FIX-24/33/47] Scheduler stepping
        if epoch <= warmup_epochs:
            warmup_sched.step()
        elif not encoder_frozen:
            # [FIX-47] Only step plateau after encoder is unfrozen
            warmup_done = True
            if not np.isnan(neg_dice_for_plateau):
                plateau_sched.step(neg_dice_for_plateau)

        if not val_metrics:
            continue

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
            f"  no-improve: {epochs_no_improve}/{EARLY_STOP_PATIENCE}"
            + (f"  ema={ema_neg_dice:.4f}" if ema_neg_dice is not None else "")
        )

        # TensorBoard
        writer.add_scalar("Epoch/train_loss",      train_loss,      epoch)
        writer.add_scalar("Epoch/val_loss",        val_loss,        epoch)
        writer.add_scalar("Epoch/lr",              current_lr,      epoch)
        writer.add_scalar("Epoch/best_threshold",  best_threshold,  epoch)
        writer.add_scalar("Epoch/presence_weight", presence_weight, epoch)
        writer.add_scalar("Epoch/encoder_frozen",  float(encoder_frozen), epoch)
        writer.add_scalar("Epoch/epochs_no_improve", epochs_no_improve, epoch)
        if ema_neg_dice is not None:
            writer.add_scalar("Epoch/ema_neg_dice", ema_neg_dice, epoch)

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

        # Mask overlay
        if vis_img is not None:
            mask_path = vis_dir / f"epoch_{epoch:04d}.png"
            save_mask_overlay(
                model, vis_img, vis_mask,
                str(mask_path), epoch, device,
                threshold=best_threshold,
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

        # Best model checkpoint [FIX-29]
        if not np.isnan(dice_mean) and dice_mean > best_dice:
            best_dice         = dice_mean
            epochs_no_improve = 0
            best_model_state  = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
            torch.save({
                "epoch"         : epoch,
                "model_state"   : model.state_dict(),
                "optimizer"     : optimizer.state_dict(),
                "dice_mean"     : dice_mean,
                "composite"     : composite,
                "best_dice"     : best_dice,
                "best_threshold": best_threshold,
                "config"        : cfg,
            }, cfg["checkpoint"])
            print(f"  ✓ Saved best model (dice_mean={best_dice:.4f}, "
                  f"composite={fmt(composite)}, "
                  f"threshold={best_threshold:.2f}) → {cfg['checkpoint']}")
        else:
            # [FIX-45] Only burn patience when encoder is unfrozen
            if not encoder_frozen:
                epochs_no_improve += 1

        # Crash checkpoint
        torch.save({
            "epoch"            : epoch,
            "model_state"      : model.state_dict(),
            "optimizer"        : optimizer.state_dict(),
            "scheduler"        : warmup_sched.state_dict(),
            "plateau_scheduler": plateau_sched.state_dict(),
            "scaler"           : scaler.state_dict() if scaler is not None else None,
            "best_dice"        : best_dice,
            "best_threshold"   : best_threshold,
            "global_step"      : global_step,
            "log_history"      : log_history,
            "epochs_no_improve": epochs_no_improve,
            "ema_neg_dice"     : ema_neg_dice,
            "encoder_frozen"   : encoder_frozen,
            "config"           : cfg,
        }, cfg["crash_checkpoint"])

        # JSON log
        log_history.append({
            "epoch"            : epoch,
            "train_loss"       : train_loss,
            "val_loss"         : val_loss,
            "best_threshold"   : best_threshold,
            "presence_weight"  : presence_weight,
            "epochs_no_improve": epochs_no_improve,
            "ema_neg_dice"     : ema_neg_dice,
            "encoder_frozen"   : encoder_frozen,
            "lr"               : current_lr,
            **{k: (float(v) if not np.isnan(float(v)) else None)
               for k, v in val_metrics.items()
               if isinstance(v, (int, float))},
        })
        with open(cfg["log_file"], "w") as f:
            json.dump(log_history, f, indent=2)

        # [FIX-36/48] Early stopping — gated by unfreeze + min_epochs [FIX-42]
        if (epochs_no_improve >= EARLY_STOP_PATIENCE
                and epoch >= MIN_EPOCHS
                and not encoder_frozen
                and not args.fast):
            print(f"\n  [Early Stop] No improvement for {EARLY_STOP_PATIENCE} "
                  f"post-unfreeze epochs. Best dice_mean={best_dice:.4f}.")
            if best_model_state is not None:
                print("  Restoring best model weights before exit.")
                model.load_state_dict(
                    {k: v.to(device) for k, v in best_model_state.items()}
                )
            break

    writer.close()
    print(f"\nTraining complete.")
    print(f"  Best dice_mean : {best_dice:.4f}")
    print(f"  Best threshold : {best_threshold:.2f}")
    print(f"  Best checkpoint: {cfg['checkpoint']}")
    print(f"  Log            : {cfg['log_file']}")
    print(f"  TensorBoard    : tensorboard --logdir runs/")
    print(f"  Mask overlays  : {cfg['vis_dir']}/epoch_XXXX.png")


if __name__ == "__main__":
    main()