"""
loss.py — Combined BCE + Dice Loss and evaluation metrics

FIXES applied:
  [FIX-3]  DiceLoss smooth: 1.0 → 1e-6. Near-zero preds now yield loss≈1.0
           instead of ≈0.0, restoring Dice gradient signal early in training.
  [FIX-7]  hausdorff_distance_2d exported so train.py validate() can import
           and call it directly per-item per-class.
  [FIX-11] hausdorff_distance_2d returns None (not 0.0) when both masks are
           empty, so callers skip the pair from the HD mean.
  [FIX-26] compute_metrics uses smooth=1.0 in the METRIC (not the loss) so
           small-structure classes don't report a misleadingly near-zero Dice.
  [FIX-32] DiceLoss smooth raised from 1e-6 → 1e-4.
           With 57% empty slices, smooth=1e-6 caused loss→0 on both-empty
           pairs, making "predict nothing" a local optimum. 1e-4 keeps a
           gradient signal flowing without materially affecting non-empty
           predictions. The metric still uses smooth=1.0 (FIX-26).
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from scipy.spatial.distance import directed_hausdorff
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Dice loss  [FIX-32] smooth=1e-4
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    [FIX-32] smooth=1e-4 (was 1e-6).
    Prevents zero-gradient trap on both-empty slice pairs (57% of dataset).
    """
    def __init__(self, smooth: float = 1e-4):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        B       = probs.shape[0]
        probs   = probs.view(B, -1)
        targets = targets.view(B, -1)
        inter   = (probs * targets).sum(dim=1)
        dice    = (2.0 * inter + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """50/50 BCE + Dice."""
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce         = nn.BCEWithLogitsLoss()
        self.dice        = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return (self.bce_weight  * self.bce(logits, targets)
              + self.dice_weight * self.dice(logits, targets))


# ---------------------------------------------------------------------------
# Hausdorff distance  [FIX-7, FIX-11]
# ---------------------------------------------------------------------------

def hausdorff_distance_2d(pred_mask: np.ndarray,
                           gt_mask:   np.ndarray):
    """
    Normalised 2-D symmetric Hausdorff distance.

    Returns
    -------
    None   Both masks empty — caller should skip this pair entirely. [FIX-11]
    1.0    Exactly one mask empty (worst case).
    float  Symmetric normalised HD in (0, 1).
    """
    if not _HAVE_SCIPY:
        return None

    pred_pts = np.argwhere(pred_mask)
    gt_pts   = np.argwhere(gt_mask)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return None

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 1.0

    h, w   = pred_mask.shape
    pred_n = pred_pts / np.array([h, w], dtype=np.float64)
    gt_n   = gt_pts   / np.array([h, w], dtype=np.float64)

    return float(max(directed_hausdorff(pred_n, gt_n)[0],
                     directed_hausdorff(gt_n, pred_n)[0]))


# ---------------------------------------------------------------------------
# Offline / backward-compat helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(preds: torch.Tensor,
                    targets: torch.Tensor,
                    threshold: float = 0.5) -> dict:
    """
    Dice + Hausdorff for a full batch.
    [FIX-11] Both-empty HD pairs skipped.
    [FIX-26] smooth=1.0 in metric.
    """
    classes    = ["large_bowel", "small_bowel", "stomach"]
    probs      = torch.sigmoid(preds).cpu().numpy()
    binary     = (probs > threshold).astype(np.uint8)
    targets_np = targets.cpu().numpy().astype(np.uint8)

    dice_scores = {c: [] for c in classes}
    hd_scores   = {c: [] for c in classes}

    for b in range(binary.shape[0]):
        for i, cls in enumerate(classes):
            p_mask, t_mask = binary[b, i], targets_np[b, i]
            inter = (p_mask & t_mask).sum()
            dsc   = (2.0 * inter + 1.0) / (p_mask.sum() + t_mask.sum() + 1.0)
            dice_scores[cls].append(float(dsc))
            hd = hausdorff_distance_2d(p_mask, t_mask)
            if hd is not None:
                hd_scores[cls].append(hd)

    results = {}
    for cls in classes:
        results[f"dice_{cls}"]      = float(np.mean(dice_scores[cls]))
        results[f"hausdorff_{cls}"] = (
            float(np.mean(hd_scores[cls])) if hd_scores[cls] else float("nan")
        )

    results["dice_mean"]      = float(np.mean([results[f"dice_{c}"] for c in classes]))
    valid_hds                 = [results[f"hausdorff_{c}"] for c in classes
                                 if not np.isnan(results[f"hausdorff_{c}"])]
    results["hausdorff_mean"] = float(np.mean(valid_hds)) if valid_hds else float("nan")

    dm, hm = results["dice_mean"], results["hausdorff_mean"]
    results["composite"] = (
        0.4 * dm + 0.6 * (1.0 - hm) if not np.isnan(hm) else float("nan")
    )
    return results