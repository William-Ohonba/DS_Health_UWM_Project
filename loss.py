"""
loss.py — Combined BCE + Dice Loss and evaluation metrics
Implements:
  - DiceLoss
  - BCEWithLogitsLoss
  - CombinedLoss (50/50 BCE + Dice as planned in Pres 3)
  - dice_coefficient() metric
  - hausdorff_distance() metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from scipy.spatial.distance import directed_hausdorff
except ImportError:
    directed_hausdorff = None


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs  = torch.sigmoid(logits)
        batch  = probs.shape[0]
        probs  = probs.view(batch, -1)
        targets = targets.view(batch, -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    50/50 BCE + Dice loss as specified in Presentation 3 Planned Improvements.
    Addresses class imbalance better than BCE alone.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce         = nn.BCEWithLogitsLoss()
        self.dice        = DiceLoss()

    def forward(self, logits, targets):
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def dice_coefficient(preds, targets, threshold=0.5, smooth=1.0):
    """
    Compute per-class Dice coefficient.
    Args:
        preds   : [B, C, H, W] raw logits or probabilities
        targets : [B, C, H, W] binary masks
    Returns:
        dict of {class_name: dice_score}
    """
    classes = ["large_bowel", "small_bowel", "stomach"]
    if preds.shape == targets.shape and preds.max() > 1:
        probs = torch.sigmoid(preds)
    else:
        probs = preds

    binary = (probs > threshold).float()
    scores = {}

    for i, cls in enumerate(classes):
        p = binary[:, i].view(-1)
        t = targets[:, i].view(-1)
        intersection = (p * t).sum()
        dsc = (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
        scores[cls] = dsc.item()

    scores["mean"] = np.mean(list(scores.values()))
    return scores


def hausdorff_distance_2d(pred_mask, gt_mask):
    """
    Compute normalized 2D Hausdorff distance between two binary masks.
    Returns 0.0 if both masks are empty.
    """
    if directed_hausdorff is None:
        return 0.0

    pred_pts = np.argwhere(pred_mask)
    gt_pts   = np.argwhere(gt_mask)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 1.0

    # Normalize coordinates by image size
    h, w    = pred_mask.shape
    pred_n  = pred_pts / np.array([h, w])
    gt_n    = gt_pts   / np.array([h, w])

    d1 = directed_hausdorff(pred_n, gt_n)[0]
    d2 = directed_hausdorff(gt_n, pred_n)[0]
    return max(d1, d2)


@torch.no_grad()
def compute_metrics(preds, targets, threshold=0.5):
    """
    Compute Dice + Hausdorff for a batch.
    Returns dict matching competition metric format.
    """
    classes = ["large_bowel", "small_bowel", "stomach"]
    probs   = torch.sigmoid(preds).cpu().numpy()
    binary  = (probs > threshold).astype(np.uint8)
    targets = targets.cpu().numpy().astype(np.uint8)

    dice_scores = {cls: [] for cls in classes}
    hd_scores   = {cls: [] for cls in classes}

    for b in range(binary.shape[0]):
        for i, cls in enumerate(classes):
            p_mask = binary[b, i]
            t_mask = targets[b, i]

            # Dice
            inter = (p_mask & t_mask).sum()
            dsc   = (2.0 * inter + 1.0) / (p_mask.sum() + t_mask.sum() + 1.0)
            dice_scores[cls].append(dsc)

            # Hausdorff
            hd = hausdorff_distance_2d(p_mask, t_mask)
            hd_scores[cls].append(hd)

    results = {}
    for cls in classes:
        results[f"dice_{cls}"]        = np.mean(dice_scores[cls])
        results[f"hausdorff_{cls}"]   = np.mean(hd_scores[cls])

    results["dice_mean"]      = np.mean([results[f"dice_{c}"]       for c in classes])
    results["hausdorff_mean"] = np.mean([results[f"hausdorff_{c}"]  for c in classes])

    # Competition composite score: 0.4 * Dice + 0.6 * (1 - Hausdorff)
    results["composite"] = (
        0.4 * results["dice_mean"] + 0.6 * (1.0 - results["hausdorff_mean"])
    )
    return results
