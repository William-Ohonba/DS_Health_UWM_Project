"""
augmentations.py — Albumentations pipeline for GI Tract segmentation

"""

import albumentations as A
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Mask-aware Random ROI Crop  [FIX-7]
# ---------------------------------------------------------------------------

class MaskAwareRandomCrop(A.DualTransform):
    """
    Crop a (crop_h × crop_w) patch biased toward non-zero mask pixels.

    Expects a 2-D binary union mask in the `mask` key of
    get_params_dependent_on_targets.  The multi-channel segmentation mask
    is passed separately as an additional_target so Albumentations applies
    the same spatial transform to it, but the anchor-point search only
    uses the 2-D union mask.

    With probability `mask_focus_prob` the crop centre is drawn from the
    non-zero region of the union mask.  Falls back to random cropping when
    the mask is completely empty or the random draw fires.
    """

    def __init__(self, height: int, width: int,
                 mask_focus_prob: float = 0.7,
                 always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.height          = height
        self.width           = width
        self.mask_focus_prob = mask_focus_prob

    @property
    def targets_as_params(self):
        return ["mask"]   # 2-D union mask from dataset.py  [FIX-7]

    def get_params_dependent_on_targets(self, params):
        mask   = params["mask"]            # guaranteed 2-D (H, W)  [FIX-7]
        img_h, img_w = mask.shape[:2]
        crop_h = min(self.height, img_h)
        crop_w = min(self.width,  img_w)

        if np.random.random() < self.mask_focus_prob:
            nonzero = np.argwhere(mask > 0)
            if len(nonzero) > 0:
                anchor = nonzero[np.random.randint(len(nonzero))]
                cy, cx = int(anchor[0]), int(anchor[1])
                y1 = int(np.clip(cy - crop_h // 2, 0, img_h - crop_h))
                x1 = int(np.clip(cx - crop_w // 2, 0, img_w - crop_w))
                return {"y_min": y1, "x_min": x1,
                        "y_max": y1 + crop_h, "x_max": x1 + crop_w}

        y1 = np.random.randint(0, max(1, img_h - crop_h + 1))
        x1 = np.random.randint(0, max(1, img_w - crop_w + 1))
        return {"y_min": y1, "x_min": x1,
                "y_max": y1 + crop_h, "x_max": x1 + crop_w}

    def apply(self, img, y_min=0, x_min=0, y_max=0, x_max=0, **params):
        return img[y_min:y_max, x_min:x_max]

    def apply_to_mask(self, mask, y_min=0, x_min=0, y_max=0, x_max=0, **params):
        return mask[y_min:y_max, x_min:x_max]

    def get_transform_init_args_names(self):
        return ("height", "width", "mask_focus_prob")


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def get_train_augmentations(img_size: int = 320) -> A.Compose:
    """
    Full augmentation pipeline for training.

    Caller (dataset.py) must supply:
        image       — 2-D centre slice (H, W) float32
        mask        — 2-D binary UNION mask (H, W) uint8  [FIX-7]
        multichan   — (H, W, C) multi-channel mask via additional_targets

    Order:
      1. Scale up 2× so random crops have context
      2. Geometric  (flip, rotate, shift-scale)
      3. Deformation (elastic, grid, optical)
      4. Mask-aware random ROI crop → target size
      5. Intensity  (brightness, noise, blur)
    """
    scaled = int(img_size * 2)

    return A.Compose([
        # 1. Scale up
        A.Resize(height=scaled, width=scaled),

        # 2. Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),

        # [FIX-33] A.Affine replaces ShiftScaleRotate.
        # translate_percent matches the old shift_limit=0.05 (±5% of image
        # dimension). scale and rotate match scale_limit=0.1 and
        # rotate_limit=15 respectively.
        # cval/cval_mask are the correct border-fill kwargs for Affine;
        # value/mask_value are not valid here and were never valid on
        # ShiftScaleRotate in albumentations >= 1.4.
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5,
        ),

        # 3. Deformation
        # [FIX-18] alpha_affine removed.
        # [FIX-34] value/mask_value → fill/fill_mask.
        A.ElasticTransform(
            alpha=120, sigma=6,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.4,
        ),
        # [FIX-34] value/mask_value → fill/fill_mask.
        A.GridDistortion(
            num_steps=5, distort_limit=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.3,
        ),
        # [FIX-18] shift_limit_x/y (already correct from prior fix).
        # [FIX-34] value/mask_value → fill/fill_mask.
        A.OpticalDistortion(
            distort_limit=0.2,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.2,
        ),

        # 4. Mask-aware crop
        MaskAwareRandomCrop(height=img_size, width=img_size,
                            mask_focus_prob=0.7, p=1.0),

        # 5. Intensity
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.4,
        ),
        # [FIX-8] std_range replaces deprecated var_limit
        A.GaussNoise(std_range=(0.04, 0.22), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),

    ],
    # multichan carries the full (H,W,C) segmentation mask through every
    # spatial transform without being used for anchor sampling  [FIX-7]
    additional_targets={"multichan": "mask"})


def get_val_augmentations(img_size: int = 320) -> A.Compose:
    """Minimal validation augmentations — resize only."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
    ], additional_targets={"multichan": "mask"})